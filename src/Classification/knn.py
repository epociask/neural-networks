from numpy.linalg import norm
from numpy import ndarray, array
from collections import Counter
from typing import Union, List, Dict, Tuple, NewType
from pandas import DataFrame
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from ast import literal_eval

Vector = NewType("Vector", List[float])
Matrix = NewType("Matrix", List[Vector])
ScoreTuple = NewType("ScoreTuple", Tuple[float])

## Made it concurrent because fuck it  
class knn:

    def __init__(self, k: int = 5):
        self.k = k 
        self.train_X = None
        self.train_Y = None 
    
    ##Transforms input data to only be of type list 
    @staticmethod
    def clean(dirty_data: list) -> List[Union[Matrix, Vector]]:
        for index, data in enumerate(dirty_data):
            if isinstance(data, DataFrame):
                dirty_data[index] = array(data).tolist()

            elif isinstance(data, ndarray):
                dirty_data[index] = data.tolist()

        return dirty_data

    ##Performs type assertions on input type 
    def assert_type(data) -> None:
        assert isinstance(data, list) or isinstance(data, ndarray) or isinstance(data, DataFrame)
            
    ##Validates input data and then assigns as class data members 
    def fit(self, train_X: Matrix, train_Y: Vector) -> None: 
        knn.assert_type(train_X)
        knn.assert_type(train_Y)
        cleaned_data  =  knn.clean([train_X, train_Y])
        self.train_X = cleaned_data[0]
        self.train_Y = cleaned_data[1]

    ##Gets maximum occuring neighbor from list of neighbors 
    @staticmethod
    def get_hypothesis(k: int, t: Vector, train_X: Matrix, train_Y: Vector) -> Union[float, int]:
            k_nearest_neighbors = knn.find_neighbors(k, t, train_X, train_Y)
            return knn.max_neighbor([e[1] for e in k_nearest_neighbors])
    
    ##Iterates over chunk of test_X matrix generating hypotheses, then sends through pipe to parent 
    @staticmethod
    def get_subset_predictions(k: int, train_X: Matrix, train_Y: Vector, partition: Matrix, partition_id: int, child_conn: Connection) -> None:
        hyps: List[Union[float, int]] = []
        for row in partition:
            hyps.append(knn.get_hypothesis(k, row, train_X, train_Y))

        child_conn.send(str((partition_id, hyps)))

    ##Chunks test_X into equal num_chunks matricies and returns as list 
    @staticmethod
    def chunkIt(m: Matrix, num_chunks: int) -> List[Matrix]:
        avg: float = len(m) / float(num_chunks)
        partitions: List[Matrix] = []
        last: float  = 0.0

        while last < len(m):
            partitions.append(m[int(last):int(last + avg)])
            last += avg

        return partitions
    
    ##Predicts all hypotheses associated w/ a testing matrix 
    def predict(self, test_X: Matrix) -> Vector:
        knn.assert_type(test_X)
        test_X = knn.clean([test_X])[0]
        hyps: List[float] = []
        num_chunks: int = 4 if len(test_X) >= 4 else 1
        
        ##chunk test_X
        chunks: List[Matrix] = knn.chunkIt(test_X, num_chunks)
        connections: List[Connection] = []
        
        ##iterate, create child process, and hold connection through pipe to child 
        for i in range(0, num_chunks):
            parent_conn, child_conn = Pipe()
            connections.append(parent_conn)
            p: Process  = Process(target=knn.get_subset_predictions, args=(self.k, self.train_X, self.train_Y, chunks[i], i, child_conn))
            p.start()

        j: int = 0
        ##wait for children to finish work 
        while j != num_chunks:
            for conn in connections:
                raw_data: str = conn.recv()

                if raw_data is not None:
                    j+=1
                    cleaned: Tuple[int, Matrix] = literal_eval(raw_data)
                    hyps.append(cleaned)


        hyps.sort(key=lambda x:x[0])
        sorted_hyps: List[Union[float, int]] = []

        for hyp in hyps:
            sorted_hyps.extend(hyp[1])

        return sorted_hyps 

    ##finds the k closest target neighbors associated w/ an input vector compared to the training matrix 
    @staticmethod
    def find_neighbors(k: int, row: Vector, train_X: Matrix, train_Y: Vector) -> Vector:
        neighbors: list = []
        for train_row in train_X:
            euclidean_distance: float = norm(array(row) - array(train_row))
            score_tuple: Tuple[float, float] = (euclidean_distance, train_Y[train_X.index(train_row)]) #distance, X val, Y val
            knn.handle_distance(k, score_tuple, neighbors)
        
        return neighbors

    ##Helper function to handle insertion to distances list
    ##Only inserts if calculated distance is less than value contained in distances  list of length K 
    @staticmethod
    def handle_distance(k: int, score_tuple: ScoreTuple, distances: List[ScoreTuple]) -> None:

        if len(distances) == k:
            if distances[k-1][0] > score_tuple[0]: #neighbor is found 
                    distances.append(score_tuple)
                    distances.sort()
                    distances.pop(k) #furthest neighbor goes bye bye 
                    return 

        else:
            distances.append(score_tuple) 
            distances.sort()



    ##returns most frequently occuring unique neighbor in list of nearest neighbors 
    @staticmethod
    def max_neighbor(nearest_neighbors: List[Union[float, int]]) -> Union[float, int]:
        counts = list(Counter(nearest_neighbors).values())
        return nearest_neighbors[counts.index(max(counts))]
