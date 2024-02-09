import random
import string

class RandomizedSelector:
    def __init__(self):
        self.generate_accurate = random.randint(0, 10) < 2
        self.dropout_ratio = 0.05
        self.randomized_weight = 0.8
        self.max_arr_length = 2**32
        self.randomization_max_val = 100
        self.generators = {"int": self.randomize_integer,
                      "float" : self.randomize_float,
                      "bool": self.randomize_boolean,
                      "string" : self.randomize_string,
                      "array" : self.randomize_array,
                      "null": self.randomize_null}
        
    def generate_parameter_value(self, parameter_type):  
        if self.generate_accurate:
            return self.generators[parameter_type]()               
        parameter_randomization = random.randint(0, self.randomization_max_val) < self.randomized_weight * self.randomization_max_val
        if parameter_randomization:
            return random.choice(list(self.generators.values()))()
        else:
            return self.generators[parameter_type]()
    
    def is_dropped(self):
        return random.randint(0, self.randomization_max_val) < self.dropout_ratio * self.randomization_max_val if not self.generate_accurate else False
        
    def randomize_integer(self):
        return random.randint(-2**32, 2**32)

    def randomize_float(self):
        return random.uniform(-2**32, 2**32)

    def randomize_boolean(self):
        return random.choice([True, False])

    def randomize_string(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 9999)))

    def randomize_array(self):
        return [random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))]

    def randomize_object(self):
        return {random.choice(string.ascii_letters): random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))}

    def randomize_null(self):
        return None
    
    def randomized_array_length(self):
        array_size = random.randint(0, 100)
        if array_size <= 95:
            return random.randint(0, 1000)
        else:
            return random.randint(0, 2**32)
