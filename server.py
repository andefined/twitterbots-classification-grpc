'''
GRPC Classification Server
'''
#!/usr/bin/env python

__version__ = '0.1.0'
__licence__ = 'MIT'
__author__ = 'Dimitris Papaevagelou'
__email__ = 'dimitris.papaevagelou@andefined.com'
__status__ = 'Production'

import time
import logging
import argparse
from concurrent import futures

import grpc
import numpy as np

from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier

import classify_pb2
import classify_pb2_grpc


# Set Logging
FORMAT = '[TBCLGRPC] %(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class Classify(classify_pb2_grpc.ClassifyServicer):
    '''
    Classify extends grpc.Servicer
    '''
    def __init__(self, model):
        self.model = read_csv(model)
        logging.info('Model: `%s` Loaded', model)

        self.x = self.model.values[:, 1:]
        self.y = self.model.values[:, 0]
        self.classes = []

        self.clf = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=8,
                                          random_state=42, max_depth=5, min_samples_leaf=5)
        self.fit()

    def fit(self):
        '''
        Set x, y dimmensions and fit the model
        '''
        fit = self.clf.fit(self.x, self.y)
        self.classes = fit.classes_
        logging.info('Forest Fitted')

    def Parse(self, data, context):
        '''
        Run the classifier (random forest)
        '''
        features = [data.followers, data.friends,
                    data.statuses, data.favorites, data.lists]
        logging.info('Features: %s', features)

        proba = self.clf.predict_proba([features])
        predict = self.classes[np.argmax(proba)]

        logging.info('Prediction -- CLASS: %s | SCORE: %s', predict, np.max(proba))

        return classify_pb2.UserClass(label=predict, score=np.max(proba))


def serve(host, port, model):
    '''
    Start GRPC Classification Server
    '''
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    classify_pb2_grpc.add_ClassifyServicer_to_server(Classify(model), server)
    server.add_insecure_port('{}:{}'.format(host, port))

    server.start()

    logging.info('GRPC Classification Server Listening on %s:%d', host, port)

    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serve GRPC Classification Server')
    parser.add_argument('--host, -h', dest='host',
                        help='GRPC Host Address', default='[::]')
    parser.add_argument('--port, -p', dest='port',
                        help='GRPC Host Port', default=50051)
    parser.add_argument('--model, -m', dest='model',
                        help='Path to Trainned Model', default='data/model.dat')
    args = parser.parse_args()

    logging.info('Starting GRPC Classification Server')
    serve(args.host, args.port, args.model)
