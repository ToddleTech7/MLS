import requests
import threading
import time

# Adjust these values to your liking:
NUM_PARALLEL_REQUESTS = 50
REQUEST_PAYLOAD = {"query": "Which animals can hover in the air?", "k": 2}

def send_request(payload):
    start_time = time.time()
    response = requests.post("http://localhost:8000/rag", json=payload)
    elapsed = time.time() - start_time
    if response.status_code == 200:
        print(f"Finished request in {elapsed:.3f}s.  Result: {response.json()['result'][:50]}...")
    else:
        print(f"Error: {response.status_code}, took {elapsed:.3f}s")


threads = []
for i in range(NUM_PARALLEL_REQUESTS):
    t = threading.Thread(target=send_request, args=(REQUEST_PAYLOAD,))
    threads.append(t)

start_all = time.time()
for t in threads:
    t.start()

for t in threads:
    t.join()
end_all = time.time()

total_time = end_all - start_all
print(f"\nSent {NUM_PARALLEL_REQUESTS} requests in parallel. Total wall time: {total_time:.3f}s.")
