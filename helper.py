from tensorflow import keras
from tensorflow.keras import layers
from keras.models import clone_model
import random
import numpy as np
import time
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  

class Darwin:

    def __init__(self):
        self.parent = Simulator(self.new_model(), 'parent')
        self.setup()
        self.step_count = 0
        self.generation_num = 0
        self.num_children = 50

    def setup(self):
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
        config.log_device_placement = True  # to log device placement (on which device the operation ran)  
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
        sess = tf.Session(config=config)  
        set_session(sess)  # set this TensorFlow session as the default session for Keras  

    
    def new_model(self):
        inputs = keras.Input(shape=(601,))
        
        x = layers.Dense(512, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="trader_bot_9000")

        # model.summary()

        return model

    def create_evolution(self, parent):
        new_model = self.new_model()
        new_weights = []

        # if self.generation_num < 5:
        #     odds = 1
        # elif self.generation_num < 10:
        #     odds = 0.5
        # elif self.generation_num < 20:
        #     odds = 0.25
        # elif self.generation_num < 35:
        #     odds = 0.15
        # elif self.generation_num < 50:
        #     odds = 0.075
        # elif self.generation_num < 100:
        #     odds = 0.025
        # else:
        #     odds = 0.02

        if self.generation_num < 5:
            odds = 0.5
        elif self.generation_num < 8:
            odds = 0.25
        elif self.generation_num < 11:
            odds = 0.15
        elif self.generation_num < 14:
            odds = 0.075
        elif self.generation_num < 17:
            odds = 0.0375
        elif self.generation_num < 21:
            odds = 0.01275
        else:
            odds = 0.01

        for layer in parent.model.get_weights():
            new_layer = []
            for weight in layer:
                new_weight = weight
                if (random.random() < odds):
                    new_weight *= random.expovariate(1)
                if (random.random() < odds):
                    new_weight *= -1
                new_layer.append(new_weight)
            new_weights.append(np.array(new_layer))
        
        new_model.set_weights(new_weights)

        return new_model

    def produce_children(self, best_children):
        self.destroy_bad_children(best_children)

        self.generation_num += 1
        potential_parents = [ self.parent ] + best_children
        potential_parents.sort(reverse=True, key=lambda child: child.money)
        new_children = []

        while len(new_children) + len(potential_parents) < self.num_children:
            new_children.append(self.create_evolution(
                random.choices(potential_parents, weights=[2**i for i in range(len(potential_parents))][::-1])[0]
            ))
        
        new_children = [ parent.model for parent in potential_parents ] + new_children
        self.children = [Simulator(model, i) for i, model in enumerate(new_children)]
    
    def sim_step(self, input_data, newestData):
        self.step_count += 1
        for child in self.children:
            child.step(input_data, newestData)
    
    def select_best(self):
        best_children = list(filter(lambda child: child.trade_count > 1, self.children))
        best_children = sorted(best_children, key=lambda child: child.money, reverse=True)[0:min(len(best_children), self.num_children//3)]
        print(str(len(best_children)) + ' children passing on genes')
        self.produce_children(best_children)
        self.parent = Simulator(best_children[0].model, 'parent')

    def protect_simulation(self, sim):
        sim.weights = sim.model.get_weights()
        
    def restore_simulation(self, sim):
        sim.model = self.new_model()
        sim.model.set_weights(sim.weights)

    def destroy_bad_children(self, children_to_keep):
        self.protect_simulation(self.parent)
        for child in children_to_keep:
            self.protect_simulation(child)
        
        keras.backend.clear_session()

        self.restore_simulation(self.parent)
        for child in children_to_keep:
            self.restore_simulation(child)

        print('bad children purged')

    def save(self, f):
        with open(f + '.darwin', 'wb') as handle:
            pickle.dump({
                'step_count': self.step_count,
                'generation_num': self.generation_num,
                'parent': self.parent.model.get_weights(),
                'children': [ child.model.get_weights() for child in self.children ]
            }, handle)

    def load(self, f):
        with open(f + '.darwin', 'rb') as handle:
            saved = pickle.load(handle)
            self.step_count = saved['step_count']
            self.generation_num = saved['generation_num']
            self.parent = Simulator(self.new_model(), 'parent')
            self.parent.model.set_weights(saved['parent'])
            self.children = []
            for i, child_weights in enumerate(saved['children']):
                model = self.new_model()
                model.set_weights(child_weights)
                self.children.append(Simulator(model, i))

class Simulator:

    def __init__(self, model, id):
        self.model = model
        self.id = id
        self.weights = 'unsaved'
        self.trade_count = 0
        self.new_sim()

        self.transaction_cost = 1

    def new_sim(self):
        self.money = 1.0
        self.last_order = 'sell'
        self.last_buy_price = 0.0
        self.last_buy_amount = 1.0
        self.step_num = 0
        self.status = ''
        self.last_prediction = 0
    
    def step(self, input_data, newestData):
        price = float(newestData['price'])
        best_bid = float(newestData['best_bid'])
        best_ask = float(newestData['best_ask'])

        self.step_num += 1
        if isinstance(input_data, str):
            order = input_data
        else:
            packaged_input = np.array([np.concatenate(([ -1.0 if self.last_order == 'sell' else 1.0 ], input_data))])
            prediction = self.model.predict(packaged_input)[0][0]
            self.last_prediction = prediction
            order = 'buy' if prediction >= 0.5 else 'sell'
        if order != self.last_order:
            self.trade_count += 1
            if order == 'buy':
                self.last_buy_price = best_ask
                self.last_buy_amount = self.money * self.transaction_cost
            if order == 'sell':
                self.money = ( best_bid / self.last_buy_price ) * self.last_buy_amount * self.transaction_cost
                self.last_buy_price = 0
            self.last_order = order
            self.status = f"{order} order"
        else:
            if order == 'buy':
                self.money = ( best_bid  / self.last_buy_price ) * self.last_buy_amount
            self.status = 'no order'
