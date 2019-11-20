import time
from lib.generator.fbm_generator import FBMExchange


start = time.time()
ex = FBMExchange()
ex.reset()
end = time.time()


print("Time: ", end-start)