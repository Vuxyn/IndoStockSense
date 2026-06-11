import requests
from bs4 import BeautifulSoup
import time
import random

def scrape_stock_news():
    """Scrapes latest stock news from multiple portals and Reddit politely."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    sources = [
        {
            "name": "CNBC Indonesia",
            "url": "https://www.cnbcindonesia.com/market",
            "type": "html"
        },
        {
            "name": "Bisnis Market",
            "url": "https://market.bisnis.com/",
            "type": "html"
        },
        {
            "name": "Kontan Investasi",
            "url": "https://investasi.kontan.co.id/",
            "type": "html"
        },
        {
            "name": "Reddit r/finansial",
            "url": "https://www.reddit.com/r/finansial/new.json?limit=10",
            "type": "reddit"
        }
    ]
    
    all_news = []
    
    for source in sources:
        print(f"[Scraper] Fetching from {source['name']}...")
        try:
            # Reddit specific User-Agent logic to avoid 429 Too Many Requests
            req_headers = headers.copy()
            if source["type"] == "reddit":
                req_headers["User-Agent"] = "python:indostocksense.bot:v1.0 (by /u/indostocksense)"
                
            response = requests.get(source['url'], headers=req_headers, timeout=10)
            
            count = 0
            if source["type"] == "html":
                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.find_all(["article", "li", "div"], limit=50) 
                
                for article in articles:
                    if count >= 3: 
                        break
                        
                    title_elem = article.find(["h2", "h3"])
                    link_elem = article.find("a")
                    
                    if title_elem and link_elem and "href" in link_elem.attrs:
                        title = title_elem.text.strip()
                        link = link_elem["href"]
                        
                        if link.startswith("/"):
                            from urllib.parse import urlparse
                            parsed_uri = urlparse(source['url'])
                            base = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
                            link = base + link
                            
                        if len(title) > 25 and not any(n['title'] == title for n in all_news):
                            all_news.append({"title": title, "url": link, "source": source['name']})
                            count += 1
                            
            elif source["type"] == "reddit":
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    if count >= 3:
                        break
                    post_data = post.get("data", {})
                    title = post_data.get("title", "")
                    link = "https://www.reddit.com" + post_data.get("permalink", "")
                    
                    if len(title) > 15 and not any(n['title'] == title for n in all_news):
                        all_news.append({"title": title, "url": link, "source": source['name']})
                        count += 1
                        
        except Exception as e:
            print(f"[Scraper] Error fetching {source['name']}: {e}")
            
        # POLITE DELAY
        if source != sources[-1]:
            delay = random.uniform(3.0, 6.0)
            print(f"[Scraper] Polite mode: Sleeping for {delay:.1f} seconds...")
            time.sleep(delay)
        
    if not all_news:
        print("[Scraper] Warning: Failed to fetch any data. Returning empty list.")
        
    return all_news
