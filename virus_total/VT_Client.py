import time
import base64 # encode URLs to b64 since its required by VT
import requests # perform http requests to VT API 
from typing import List, Dict, Union, Optional

class VT_Client:
    """
    A lightweight VirusTotal client that fetches native reputation scores 
    for URLs, Domains, and IPs. Handles missing artifact categories seamlessly.
    """
    
    BASE_URL = "https://www.virustotal.com/api/v3"

    def __init__(self, api_key: str, delay_seconds: int = 15):
        self.headers = {"x-apikey": api_key}
        self.delay = delay_seconds # 15 seconds delay as the policy of free API 

    def _encode_url_for_vt(self, url: str) -> str:
        """VirusTotal v3 requires URLs to be base64 url-safe encoded without '=' padding."""
        return base64.urlsafe_b64encode(url.encode()).decode().strip("=")
# convert string to bytes -> B64 encode -> decode bytes to string -> remove padding "="
# encode works only with bytes, urlsafe_b64encode() replace +,/ with -,_ to avoid confusion, .decode() convert bytes to str, remove = to avoid causing API call issues.
    def _get_reputation(self, endpoint: str, identifier: str) -> Union[int, str]:
        """
        Hits the VT API and safely extracts the raw 'reputation' integer.
        Handles rate limits dynamically.
        """
        api_url = f"{self.BASE_URL}/{endpoint}/{identifier}"
        
        while True:
            try:
                response = requests.get(api_url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json() # convert str to python dict to use key:value
                    return data.get("data", {}).get("attributes", {}).get("reputation", 0)
                # extract the native reputation score. .get() with defaults to avoid keyError if any level is missing
                elif response.status_code == 429:
                    print(f"[-] Rate limit hit. Sleeping for {self.delay} seconds...")
                    time.sleep(self.delay)
                    continue
                
                elif response.status_code == 404:
                    return "Not Found. Check another intelligence resource"
                    
                elif response.status_code in [401, 403]:
                    return "Auth/Permission Error"
                    
                else:
                    return f"API Error {response.status_code}"
                    
            except requests.RequestException as e:
                return f"Network Error: {str(e)}"

    def get_reputations(self, 
                        urls: Optional[List[str]] = None, 
                        domains: Optional[List[str]] = None, 
                        ips: Optional[List[str]] = None) -> Dict[str, Union[int, str]]:
        """
        Processes available artifacts. If one category is missing, it simply 
        skips to the next without aborting the process.
        """
        # Safely convert None to empty lists
        urls = urls or []
        domains = domains or []
        ips = ips or []

        results = {}

        # Early exit ONLY if absolutely every category is empty
        if not any([urls, domains, ips]):
            print("No artifacts found across any category.")
            return results

        # 1. Process URLs (Skips seamlessly if urls list is empty)
        if urls:
            for url in urls:
                if url:  # Protects against empty strings like [""]
                    vt_id = self._encode_url_for_vt(url)
                    rep = self._get_reputation("urls", vt_id)
                    results[url] = rep # use the original url as key
                    print(f"{url} : {rep}")
                    time.sleep(self.delay)

        # 2. Process Domains (Skips seamlessly if domains list is empty)
        if domains:
            for domain in domains:
                if domain:
                    rep = self._get_reputation("domains", domain)
                    results[domain] = rep
                    print(f"{domain} : {rep}")
                    time.sleep(self.delay)

        # 3. Process IPs (Skips seamlessly if ips list is empty)
        if ips:
            for ip in ips:
                if ip:
                    rep = self._get_reputation("ip_addresses", ip)
                    results[ip] = rep
                    print(f"{ip} : {rep}")
                    time.sleep(self.delay)

        return results

# ==========================================
# Example Usage Scenarios
# ==========================================
if __name__ == "__main__":
    API_KEY = "PUT YOUR API HERE"
    client = VT_Client(api_key=API_KEY, delay_seconds=0) # Set to 0 just for fast local testing
    
    print("Processing VT Client... ")
    client.get_reputations(
        urls=[""], 
        domains=[], 
        ips=[]
    )
 

