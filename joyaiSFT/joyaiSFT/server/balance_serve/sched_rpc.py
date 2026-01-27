from datetime import datetime
import os
from typing import Optional
import zmq
import pickle
import threading
import torch.multiprocessing as mp
import sys
current_file_path = os.path.abspath(__file__)
import pickle
import argparse
from joyaiSFT.server.balance_serve.settings import sched_ext, create_sched_settings
if mp.get_start_method(allow_none=True) is None:
    print('set start method')
    mp.set_start_method('spawn')
else:
    print(f'start method already set to {mp.get_start_method(allow_none=True)}')

class SchedulerServer:

    def __init__(self, settings, main_args):
        self.sched = sched_ext.create_scheduler(settings)
        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)
        print(f'sched zmq rpc server on port {main_args.sched_port}')
        self.frontend.bind(f'tcp://*:{main_args.sched_port}')
        self.backend = self.context.socket(zmq.DEALER)
        self.backend.bind('inproc://backend')

    def run_scheduler(self):
        self.sched.run()

    def stop_scheduler(self):
        self.sched.stop()

    def start_proxy(self):
        zmq.proxy(self.frontend, self.backend)

    def worker_routine(self):
        worker = self.context.socket(zmq.REP)
        worker.connect('inproc://backend')
        while True:
            try:
                message = worker.recv()
                data = pickle.loads(message)
                method = data.get('method')
                params = data.get('params', {})
                if method == 'add_query':
                    query_add = params.get('query')
                    query_id = self.sched.add_query(query_add)
                    response = {'status': 'ok', 'query_id': query_id}
                    worker.send(pickle.dumps(response))
                elif method == 'cancel_query':
                    query_id = params.get('query_id')
                    self.sched.cancel(query_id)
                    response = {'status': 'ok'}
                    worker.send(pickle.dumps(response))
                elif method == 'update_last_batch':
                    updates = params.get('updates')
                    batch_todo = self.sched.update_last_batch(updates)
                    response = {'status': 'ok', 'batch_todo': batch_todo}
                    worker.send(pickle.dumps(response))
                elif method == 'get_inference_context':
                    inference_context = self.sched.get_inference_context()
                    data = {'k_cache': inference_context.k_cache, 'v_cache': inference_context.v_cache}
                    print(f'Serializing KVCache')
                    data['k_cache'] = [mp.reductions.reduce_tensor(t) for t in data['k_cache']]
                    data['v_cache'] = [mp.reductions.reduce_tensor(t) for t in data['v_cache']]
                    response = {'status': 'ok', 'inference_context': data}
                    worker.send(pickle.dumps(response))
                else:
                    response = {'status': 'error', 'message': 'Unknown method'}
                    worker.send(pickle.dumps(response))
            except Exception as e:
                response = {'status': 'error', 'message': str(e)}
                worker.send(pickle.dumps(response))

    def start_rpc_service(self):
        try:
            print('Scheduler RPC service is running...')
            threading.Thread(target=self.run_scheduler, daemon=True).start()
            for _ in range(10):
                threading.Thread(target=self.worker_routine, daemon=True).start()
            self.start_proxy()
        except KeyboardInterrupt:
            print('Shutting down scheduler RPC service...')
            self.stop_rpc_service()

    def stop_rpc_service(self):
        self.stop_scheduler()
        self.frontend.close()
        self.backend.close()
        self.context.term()

def start_server(settings, main_args):
    server = SchedulerServer(settings, main_args)
    server.start_rpc_service()

class SchedulerClient:

    def __init__(self, sched_port):
        address = f'tcp://localhost:{sched_port}'
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)
        print(f'Connected to server at {self.address}')

    def __del__(self):
        self.socket.close()
        self.context.term()

    def send_request(self, method, params=None):
        if params is None:
            params = {}
        request = {'method': method, 'params': params}
        self.socket.send(pickle.dumps(request))
        response = self.socket.recv()
        response = pickle.loads(response)
        if response.get('status') == 'ok':
            return response
        else:
            raise Exception(f"Error from server: {response.get('message')}")

    def add_query(self, query):
        response = self.send_request('add_query', {'query': query})
        return response.get('query_id')

    def cancel_query(self, query_id):
        self.send_request('cancel_query', {'query_id': query_id})

    def update_last_batch(self, updates):
        response = self.send_request('update_last_batch', {'updates': updates})
        return response.get('batch_todo')

    def rebuild_inferece_context(self, response):
        data = response.get('inference_context')
        inference_context = sched_ext.InferenceContext()
        print('Rebuilding kvcache')
        inference_context.k_cache = [fn(*args) for fn, args in data['k_cache']]
        inference_context.v_cache = [fn(*args) for fn, args in data['v_cache']]
        return inference_context

    def get_inference_context_raw(self):
        response = self.send_request('get_inference_context')
        return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'rb') as f:
        main_args = pickle.load(f)
    settings = create_sched_settings(main_args)
    start_server(settings, main_args)