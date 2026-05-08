import pandas as pd
import asyncio
import aiohttp
from pathlib import Path
from tqdm.asyncio import tqdm
import re
import urllib.parse
import os

DATA_DIR = Path.cwd().parent / "data"
ITEMS_PATH = DATA_DIR / "items.csv"
OUTPUT_PATH = DATA_DIR / "full_item_dataset.csv"

# Put multiple keys here. If you only have one, just leave one in the list.
# You can add more later if the script stops!
API_KEYS = []
current_key_idx = 0  # Tracks which key we are using

def get_current_key_param():
    """Returns the URL parameter for the active API key."""
    if not API_KEYS or API_KEYS[0] == "YOUR_KEY_1":
        return ""
    return f"&key={API_KEYS[current_key_idx]}"

def extract_valid_isbns(raw_isbn_str):
    if pd.isna(raw_isbn_str) or str(raw_isbn_str).strip() == "":
        return []
    raw_codes = str(raw_isbn_str).split(';')
    valid_isbns = [re.sub(r'[\s-]', '', code) for code in raw_codes]
    return [code for code in valid_isbns if len(code) in [10, 13]]

async def fetch_from_google(session, base_url):
    global current_key_idx
    for attempt in range(3):
        url = base_url + get_current_key_param()
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if "items" in data and len(data["items"]) > 0:
                        volume_info = data["items"][0].get("volumeInfo", {})
                        
                        genres = ", ".join(volume_info.get("categories", []))
                        description = volume_info.get("description", "")
                        
                        # NEW: Extract numerical features
                        page_count = volume_info.get("pageCount", None)
                        avg_rating = volume_info.get("averageRating", None)
                        
                        return genres, description, page_count, avg_rating
                    return None, None, None, None 
                
                elif response.status == 429:
                    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    if current_key_idx == 0:
                        await asyncio.sleep(60)
        except Exception:
            pass
        await asyncio.sleep(1)
    return None, None, None, None

async def fetch_book_data(session, row, semaphore):
    item_id = row['i']
    
    # Extract values
    title = row.get('Title', '')
    author = row.get('Author', '')
    raw_isbn = row.get('ISBN Valid', '')
    
    isbns = extract_valid_isbns(raw_isbn)
    genres, description = None, None

    async with semaphore:
        # ---------------------------------------------------------
        # ATTEMPT 1: Search by ISBN
        # ---------------------------------------------------------
        for isbn in isbns:
            url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
            genres, description, page_count, avg_rating = await fetch_from_google(session, url)
            if genres or description:
                break 
        
        # ---------------------------------------------------------
        # ATTEMPT 2: Strict Title + Author (No Publisher)
        # ---------------------------------------------------------
        if not genres and not description and pd.notna(title) and str(title).strip():
            
            # Basic text cleaning to remove special characters that break the API
            clean_title = re.sub(r'[^\w\s]', '', str(title).strip())
            clean_title_url = urllib.parse.quote(clean_title)
            
            query_strict = f"intitle:{clean_title_url}"
            
            if pd.notna(author) and str(author).strip():
                clean_author = re.sub(r'[^\w\s]', '', str(author).strip())
                clean_author_url = urllib.parse.quote(clean_author)
                query_strict += f"+inauthor:{clean_author_url}"
                
            url_strict = f"https://www.googleapis.com/books/v1/volumes?q={query_strict}"
            genres, description, page_count, avg_rating = await fetch_from_google(session, url_strict)
            
            # ---------------------------------------------------------
            # ATTEMPT 3: Loose Keyword Search (Mimics Web Search)
            # ---------------------------------------------------------
            if not genres and not description:
                # Just mash the title and author together as general keywords
                query_loose = clean_title_url
                if pd.notna(author) and str(author).strip():
                    query_loose += f"+{clean_author_url}"
                
                url_loose = f"https://www.googleapis.com/books/v1/volumes?q={query_loose}"
                genres, description, page_count, avg_rating = await fetch_from_google(session, url_loose)
        
        await asyncio.sleep(0.15)

    return {
        "item_id": item_id,
        "genres": genres if genres else "",
        "summary": description if description else "",
        "page_count": page_count,
        "average_rating": avg_rating,
        "api_found": bool(genres or description)
    }

async def main():
    print("Loading items.csv")
    items_df = pd.read_csv(ITEMS_PATH)
    
    # CHECKPOINTING: Read existing data so we don't start from scratch
    if os.path.exists(OUTPUT_PATH):
        enriched_data = pd.read_csv(OUTPUT_PATH)
        completed_ids = set(enriched_data['item_id'].values)
        print(f"Found existing save file! {len(completed_ids)} books already fetched.")
    else:
        completed_ids = set()
        # Create an empty CSV with headers to start appending to
        pd.DataFrame(columns=["item_id", "genres", "summary", "api_found"]).to_csv(OUTPUT_PATH, index=False)
        enriched_data = pd.DataFrame()

    # Filter out items we already have
    items_to_fetch = items_df[~items_df['i'].isin(completed_ids)]
    print(f"Remaining books to fetch: {len(items_to_fetch)}")

    if len(items_to_fetch) == 0:
        print("All books have been fetched! You are ready to move on.")
        return

    semaphore = asyncio.Semaphore(5)
    
    # BATCH PROCESSING: Save progress every 200 items
    batch_size = 100
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(items_to_fetch), batch_size):
            batch = items_to_fetch.iloc[i : i + batch_size]
            
            print(f"\nProcessing batch {i} to {i + len(batch)}")
            tasks = [fetch_book_data(session, row, semaphore) for _, row in batch.iterrows()]
            
            # Run the batch
            results = await tqdm.gather(*tasks)
            
            # Save the batch immediately to the CSV
            batch_df = pd.DataFrame(results)
            batch_df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
            
            # Update our completed IDs in memory just in case
            completed_ids.update(batch_df['item_id'].values)

    print("\nFetch complete! Merging enriched data with original items")
    
    # Final merge just to give you a clean dataframe at the end
    full_enriched_data = pd.read_csv(OUTPUT_PATH)
    final_df = items_df.merge(full_enriched_data, left_on='i', right_on='item_id', how='left')
    final_df.drop(columns=['item_id'], inplace=True)
    
    final_csv_path = DATA_DIR / "final_enriched_items.csv"
    final_df.to_csv(final_csv_path, index=False)
    
    print(f"Success! Merged data saved to {final_csv_path}")
    print(f"Total APIs hits found: {final_df['api_found'].sum()}/{len(final_df)}")

await main()