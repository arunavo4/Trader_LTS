import random as rand

import numpy as np
import pandas as pd
from stochastic.noise import GaussianNoise

from lib.generator.stochastic.stoch_gen import *


def get_price_data(model_type, mp):
    if model_type.upper() == "GBM":
        m = geometric_brownian_motion_levels(mp)
    elif model_type.upper() == "HESTON":
        m = heston_model_levels(mp)[0]
    elif model_type.upper() == "MERTON":
        m = geometric_brownian_motion_jump_diffusion_levels(mp)
    elif model_type.upper() == "COX":
        m = cox_ingersoll_ross_levels(mp)
    elif model_type.upper() == "ORNSTEIN UHLENBECK":
        m = ornstein_uhlenbeck_levels(mp)
    else:
        m = geometric_brownian_motion_levels(mp)

    return m


def get_model_params(param_type, base_price, t_gen, delta):
    if param_type == "Super":
        params = ModelParameters(
            all_s0=base_price,
            all_r0=0.5,
            all_time=t_gen,
            all_delta=delta,
            all_sigma=random.uniform(0.1, 0.8),
            gbm_mu=random.uniform(-0.3, 0.6),
            jumps_lamda=random.uniform(0.0071, 0.6),
            jumps_sigma=random.uniform(-0.03, 0.04),
            jumps_mu=random.uniform(-0.2, 0.2),
            cir_a=3.0,
            cir_mu=0.5,
            cir_rho=0.5,
            ou_a=3.0,
            ou_mu=0.5,
            heston_a=random.uniform(1, 5),
            heston_mu=random.uniform(0.156, 0.693),
            heston_vol0=0.06125
        )
    else:
        params = ModelParameters(
            all_s0=base_price,
            all_r0=0.5,
            all_time=t_gen,
            all_delta=delta,
            all_sigma=0.125,
            gbm_mu=0.058,
            jumps_lamda=0.00125,
            jumps_sigma=0.001,
            jumps_mu=-0.2,
            cir_a=3.0,
            cir_mu=0.5,
            cir_rho=0.5,
            ou_a=3.0,
            ou_mu=0.5,
            heston_a=0.25,
            heston_mu=0.35,
            heston_vol0=0.06125
        )


def get_delta(time_frame):
    if time_frame == "1sec":
        return 1 / (252 * 24 * 60 * 60)
    elif time_frame == "1min":
        return 1 / (252 * 24 * 60)
    elif time_frame == "15min":
        return 1 / (252 * 24 * 4)
    elif time_frame == "hourly":
        return 1 / (252 * 24)
    elif time_frame == "daily":
        return 1 / 252
    elif time_frame == "monthly":
        return 1 / 12


class STOCHExchange:
    """A simulated instrument exchange, in which the price history is based off a
    *   Geometric Brownian Motion
    *   The Merton Jump-Diffusion Model
    *   The Heston Stochastic Volatility Model
    *   Cox Ingersoll Ross (CIR)
    *   Ornstein Uhlebneck stochastic process
    """

    def __init__(self, **kwargs):
        self._delta = None
        self._base_price = kwargs.get('base_price', np.random.randint(500, 1000))
        self._base_volume = kwargs.get('base_volume', 1)
        self._start_date = kwargs.get('start_date', '2010-01-01')
        self._start_date_format = kwargs.get('start_date_format', '%Y-%m-%d')
        self._times_to_generate = kwargs.get('times_to_generate', 2700000)
        self._model_type = kwargs.get('model_type', rand.choice(["GBM", "HESTON", "COX", "ORNSTEIN UHLENBECK"]))
        self._param_type = kwargs.get('param_type', rand.choice(["Default", "Super"]))
        self._timeframe = kwargs.get('timeframe', '1min')

    def _generate_price_history(self):
        prices = get_price_data(self._model_type,
                                get_model_params(self._param_type, self._base_price, self._times_to_generate,
                                                 get_delta(self._timeframe)))
        volume_gen = GaussianNoise(t=self._times_to_generate)

        start_date = pd.to_datetime(self._start_date, format=self._start_date_format)

        volumes = volume_gen.sample(self._times_to_generate)

        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame(
            [], columns=['date', 'volume'], dtype=float)

        price_frame['date'] = pd.date_range(
            start=start_date, periods=self._times_to_generate, freq="S")
        price_frame['price'] = abs(prices)

        volume_frame['date'] = price_frame['date'].copy()
        volume_frame['volume'] = abs(volumes)

        price_frame.set_index('date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='s', origin=start_date)

        volume_frame.set_index('date')
        volume_frame.index = pd.to_datetime(volume_frame.index, unit='s', origin=start_date)

        data_frame = price_frame['price'].resample(self._timeframe).ohlc()
        data_frame['volume'] = volume_frame['volume'].resample(self._timeframe).sum()

        self.data_frame = data_frame.astype(np.float16)

    def reset(self):
        self._generate_price_history()
