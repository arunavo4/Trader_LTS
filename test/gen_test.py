import time
from lib.generator.static_generator import StaticExchange

ex = StaticExchange()

start = time.time()
ex.reset()
end = time.time()


print("Time: ", end-start)

print(ex.data_frame.head())
