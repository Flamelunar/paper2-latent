import concurrent.futures,time
import hybrid_eval_20261010 as h
h.status('deepseek_our_parallel_evaluation',judges=['selene8b','qwen3_14b'],gpu=1)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 tasks=[pool.submit(h.evaluate,'deepseek14b','our_direct',j,1) for j in ['selene8b','qwen3_14b']]
 while not all(t.done() for t in tasks):
  h.publish('deepseek14b');time.sleep(10)
 for t in tasks:t.result()
h.audit('deepseek14b','our_direct');h.publish('deepseek14b')
h.main()
