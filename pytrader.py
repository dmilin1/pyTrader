import asyncio
import websockets
import json
import math
from datetime import datetime
import time
import numpy as np
import multiprocessing as mp
import queue
from blessed import Terminal

subscription = json.dumps({
    "type": "subscribe",
    "product_ids": [
        "BTC-USD",
    ],
    "channels": [
        "ticker"
    ]
})

def get_entire_queue(q):
    items = [q.get()]
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    return [ [ item[0] for item in items ], items[-1][1] ]

def initialize_input(q):
    items = []
    while len(items) < 100:
        items.append(q.get())
    return [ [ item[0] for item in items ], items[-1][1] ]

def load_next_inputs(q, input_data):
    data, newest = get_entire_queue(q)
    return [(input_data + data)[-100:], newest]

# def repair_keras_session(protected):
#     from helper import Darwin, Simulator
#     from tensorflow import keras

#     saved = []

#     for obj in protected:
#         if isinstance(obj, Simulator):
            

#     keras.backend.clear_session()



def train_net(q):
    from helper import Darwin, Simulator

    term = Terminal()

    input_data, newest = initialize_input(q)

    darwin = Darwin()

    # darwin.load('test')
    # darwin.step_count = 0
    # darwin.generation_num = 0
    darwin.produce_children([])


    best_parent_sim = Simulator(darwin.parent.model, 'best_parent_sim')
    true_start_price = None

    steps_per_gen = 1000

    while True:
        start_price = None

        for i in range(steps_per_gen):
            input_data, newest = load_next_inputs(q, input_data)
            current_price = float(newest['price'])
            if start_price is None:
                start_price = current_price
            if true_start_price is None:
                true_start_price = current_price
            
            formatted_input = np.ndarray.flatten(np.array(input_data))
            darwin.sim_step(formatted_input, newest)
            
            if darwin.generation_num > 5:
                good_children = darwin.children[0:len(darwin.children)//3]
                most_popular_order = 'buy' if sum(child.last_order == 'buy' for child in good_children)/len(good_children) >= 0.5 else 'sell'
                best_parent_sim.step(most_popular_order, newest)
            
            the_str = term.clear + f"generation {darwin.generation_num} - step {darwin.step_count % steps_per_gen}/{steps_per_gen} - market_value {current_price/start_price} - {len(list(filter(lambda child: child.money > current_price/start_price, darwin.children)))} children beating market \n"
            the_str += f"\n\n{'#'.rjust(3, ' ')}: {'money'.ljust(20, ' ').rjust(25, ' ')}{'last trade'.ljust(15, ' ').rjust(20, ' ')}{'> market'.rjust(10, ' ')}{'prediction'.ljust(15, ' ').rjust(20, ' ')}{'last order'.ljust(10, ' ').rjust(15, ' ')}{'trade ct'.ljust(10, ' ').rjust(15, ' ')}"
            the_str += "\n"
            for child in darwin.children:
                the_str += f"\n{str(child.id).rjust(3, ' ')}: {str(child.money).ljust(20, ' ').rjust(25, ' ')}{child.status.ljust(15, ' ').rjust(20, ' ')}{('X' if child.money > current_price/start_price else '').rjust(10, ' ')}{str(child.last_prediction).ljust(15, ' ').rjust(20, ' ')}{child.last_order.ljust(10, ' ').rjust(15, ' ')}{str(child.trade_count).ljust(10, ' ').rjust(15, ' ')}"
            the_str += "\n\n\nmultigenerational best parent trade count: ".ljust(60, ' ') + str(best_parent_sim.trade_count)
            the_str += "\nmultigenerational best parent price: ".ljust(60, ' ') + str(best_parent_sim.money)
            the_str += "\nmultigenerational market price: ".ljust(60, ' ') + str(current_price/true_start_price)
            print(the_str)

        darwin.select_best()
        best_parent_sim.model = darwin.parent.model

        print('saving model')
        darwin.save('test')

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

async def connect(uri, q):
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(subscription)
                prev_time = time.time()
                while True:
                    d = json.loads(await asyncio.wait_for(websocket.recv(), timeout=60))
                    if d['type'] == 'ticker':
                        price, low_24h, high_24h, best_bid, best_ask, side, ticker_time = (
                            float(d['price']),
                            float(d['low_24h']),
                            float(d['high_24h']),
                            float(d['best_bid']),
                            float(d['best_ask']),
                            d['side'],
                            d['time']
                        )
                        data = [
                            # price from -1 (24h low) to 1 (24h high)
                            ((price - low_24h) / (high_24h - low_24h)) * 2 - 1,
                            # price from -1 (best bid) to 1 (best ask)
                            ((price - best_bid) / (best_ask - best_bid)) * 2 - 1,
                            # sigmoid of delta between 24h low and high as ratio of 24 hr high
                            # -1 to 1 is small to large difference between 24 hr low and high
                            (sigmoid(10 * (high_24h - low_24h) / high_24h)) * 2 - 1,
                            # sigmoid of delta between best bid and best ask as ratio of best ask
                            # -1 to 1 is small to large difference between best bid and best ask
                            (sigmoid(100 * (best_ask - best_bid) / best_ask)) * 2 - 1,
                            # 1 if last order was buyer, -1 if seller
                            1.0 if side == 'buy' else -1.0,
                            # sigmoid of time since previous item in list
                            sigmoid((datetime.strptime(ticker_time, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime(1970, 1, 1)).total_seconds() - prev_time)
                        ]
                        prev_time = (datetime.strptime(ticker_time, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime(1970, 1, 1)).total_seconds()
                        q.put([data, d])
        except Exception as e:
            print(e)
            print('Websocket failure. Will retry in 10 seconds.')
            time.sleep(10)



def data_collector(q):
    asyncio.get_event_loop().run_until_complete(
        connect('wss://ws-feed.pro.coinbase.com', q)
    )

if __name__ == '__main__':
    q = mp.Queue()
    collector = mp.Process(target=data_collector, args=(q, ))
    collector.start()

    learner_q = mp.Queue()
    learner = mp.Process(target=train_net, args=(q, ))
    learner.start()