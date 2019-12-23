import time
from lib.generator.static_generator import StaticExchange

config = {
    "initial_balance": 10000,
    "enable_env_logging": False,
    "observation_window": 84,
    "frame_stack_size": 4,
    "use_leverage": False,
    "market": 'us_mkt',
}

ex = StaticExchange(config)

start = time.time()
ex.reset()
end = time.time()


print("Time: ", end-start)
print(ex.data_frame.shape)
print(ex.data_frame.head())
print(ex.data_frame.tail())
