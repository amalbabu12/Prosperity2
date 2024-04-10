from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math
import numpy as np
import operator
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class StaticTrader:
    limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    order_book = {'AMETHYSTS': {'BUY': [], 'SELL': []}, 'STARFRUIT': {'BUY': [], 'SELL': []}}


    def put_order(product, price, vol, order_lst):
        trade_made = 'SELL'
        if vol > 0:
            trade_made = 'BUY'

        # tuples are organized: (price, volume)
        trade_to_search = 'SELL' if trade_made == 'BUY' else 'BUY'
        for price, volume in StaticTrader.order_book[product][trade_to_search]:
            volume_to_trade = min(volume, vol)



        StaticTrader.order_book[product][trade_made].append((price, vol))   
        order_lst.append(Order(product, price, vol))

    def marketmake(product, tradeMade, acceptablePrice, volume, orderList):
        quantity = int((StaticTrader.limits[product] // math.sqrt(StaticTrader.limits[product])) * 2)
        if tradeMade == "BUY":
            less = int((volume+quantity-1)//quantity)
            for i in range(int(acceptablePrice) - 2, int(acceptablePrice) - 2 - less, -1):
                vol = quantity if volume >= quantity else volume
                print("BUY", str(-vol) + "x", i)
                StaticTrader.put_order(product, i, vol, orderList)
                #orderList.append(Order(product, i, vol))
                volume -= vol
        elif tradeMade == "SELL":
            less = int((volume+quantity-1)//quantity)
            for i in range(int(acceptablePrice) + 2, int(acceptablePrice) + 2 + less):
                vol = quantity if volume >= quantity else volume
                print("SELL", str(vol) + "x", i)
                StaticTrader.put_order(product, -i, vol, orderList)
                # orderList.append(Order(product, i, -vol))
                volume -= vol
        else:
            return None
    
    def get_product_expected_price(state, product):
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        max_buy = max(buy_orders.keys())
        min_ask = min(sell_orders.keys())
                
        return ((min_ask + max_buy)/2, max_buy, min_ask)

    def get_max_min_vols(state, product):
        pos = state.position.get(product, 0)
        limit = StaticTrader.limits[product]
        max_buy = limit - pos
        max_sell = abs(-limit - pos)
        return max_buy, max_sell
    
    def do_order_price(bot_orders, operator, max_vol, acceptable_price, trade_made, product, order_lst, limit):
        reverse = False
        if trade_made == "SELL":
            reverse = True
        tradeHappened = False
        orders_sorted = sorted(bot_orders.keys(), reverse = reverse)
        all_prices = []
        for prices in orders_sorted:
            if operator(prices, acceptable_price):
                volume = abs(bot_orders[prices])
                vol_to_trade = min(volume, max_vol)
                max_vol -= vol_to_trade
                # In case the lowest ask is lower than our fair value,
                # This presents an opportunity for us to buy cheaply
                # The code below therefore sends a BUY order at the price level of the ask,
                # with the same quantity
                # We expect this order to trade with the sell order
                # print(trade_made, str(vol_to_trade) + "x", prices)
                if trade_made == "BUY":
                    StaticTrader.put_order(product, prices, vol_to_trade, order_lst)
                    #order_lst.append(Order(product, prices, vol_to_trade))
                    tradeHappened = True
                elif trade_made == "SELL":
                    StaticTrader.put_order(product, prices, -vol_to_trade, order_lst)
                    #order_lst.append(Order(product, prices, -vol_to_trade))
                    tradeHappened = True
                all_prices.append(prices)
            else: 
                break
            if max_vol <= 0:
                break
                    
        if not tradeHappened:
          StaticTrader.marketmake(product=product, tradeMade=trade_made, acceptablePrice=acceptable_price, volume=max_vol, orderList=order_lst)
          return None

        else:
            return all_prices

    def do_order_volume(bot_orders, max_vol, trade_made, product, order_lst):
        reverse = False
        if trade_made == "SELL":
            reverse = True
        tradeHappened = False
        orders_sorted = sorted(bot_orders.keys(), reverse = reverse)
        for prices in orders_sorted:
            # if operator(prices, acceptable_price):
            volume = abs(bot_orders[prices])
            vol_to_trade = min(volume, max_vol)
            # print("VOLUME TO TRADE: ", vol_to_trade)
            if vol_to_trade <= 0:
                break
            max_vol -= vol_to_trade
            # In case the lowest ask is lower than our fair value,
            # This presents an opportunity for us to buy cheaply
            # The code below therefore sends a BUY order at the price level of the ask,
            # with the same quantity
            # We expect this order to trade with the sell order
            tradeHappened = True

            # print(trade_made, str(vol_to_trade) + "x", prices)
            if trade_made == "BUY":
                StaticTrader.put_order(product, prices, vol_to_trade, order_lst)
                #order_lst.append(Order(product, prices, vol_to_trade))

            elif trade_made == "SELL":
                StaticTrader.put_order(product, prices, -vol_to_trade, order_lst)
                #order_lst.append(Order(product, prices, -vol_to_trade))

            if max_vol <= 0:
                break
        return tradeHappened

    def do_midpoint(sell_orders, buy_orders):
        return (min(sell_orders) + max(buy_orders)) / 2

class MeanReversion:

    def __init__(self, window_size: int, z_thresh: int, product : str):
        self.rolling_buys = list()
        self.rolling_asks = list()
        self.WINDOW_SIZE = window_size
        self.Z_THRESH = z_thresh
        self.product = product
        self.limit = StaticTrader.limits[product]

    # 
    def rolling_mean(self):
        return np.array(self.rolling_window[-self.WINDOW_SIZE - 1:]).mean()
    
    
    def z_score(self, x: float):
        last_window = np.array(self.rolling_window[-self.WINDOW_SIZE - 1: -1])

        return (x - last_window.mean())/last_window.std()
    
    def last_window(self, arr):
        last_window = np.array(arr[-self.WINDOW_SIZE - 1: -1])
        return last_window

    def match_orders(self, fair_bid, fair_ask, order_depth, pos):
        orders = []
        buy_pos = pos
        sell_pos = pos
        ask_stack = list(order_depth.sell_orders.items())[::-1]
        bid_stack = list(order_depth.buy_orders.items())[::-1]
        while ask_stack:
            ask, quantity = ask_stack.pop()
            q = min(-quantity, 20 - buy_pos)
            if ask < fair_bid and q > 0:
                buy_pos += q
                orders.append(Order(self.product, ask, q))
            else:
                break

        while bid_stack:
            bid, quantity = bid_stack.pop()
            q = max(-quantity, -20 - sell_pos)
            if bid > fair_ask and q < 0:
                sell_pos += q
                orders.append(Order(self.product, bid, q))
            else:
                break
        return (orders, 
                bid_stack,
                ask_stack,
                buy_pos,
                sell_pos)

    def make_am_orders(self, state):
        orders = []
        order_depth = state.order_depths[self.product]
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]
        self.rolling_buys.append(best_ask)
        self.rolling_asks.append(best_bid)
        if len(self.rolling_buys) >= self.WINDOW_SIZE:
            # buy_window = self.last_window(self.rolling_buys)
            # ask_window = self.last_window(self.rolling_asks)
            price = 10000
            orders, remaining_bids, remaining_asks, buy_pos, sell_pos = self.match_orders(price, price, order_depth, state.position.get(self.product, 0))
            next_bid = remaining_bids[-1][0] if remaining_bids else price - 2
            next_ask = remaining_asks[-1][0] if remaining_asks else price + 2
        
            if sell_pos > -20:
                offer_price = next_ask - 1 if (next_ask - 1  > price) else price + 1
                orders.append(Order(self.product, offer_price, -20 - sell_pos))
            if buy_pos < 20:
                offer_price = next_bid + 1 if (next_bid + 1  < price) else price - 1
                orders.append(Order(self.product, 10000 - 1, 20 - buy_pos))

        # if position < 0:
        #     orders.append(Order(self.product, price - 1, -position))
        # elif position > 0:
        #     orders.append(Order(self.product, price + 1, -position))



        return orders
    

    # STARFRUIT
    # acceptable price - , spread -
    def make_sf_orders(self, state):
        orders = []
        order_depth = state.order_depths[self.product]
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]
        self.rolling_buys.append(best_ask)
        self.rolling_asks.append(best_bid)
        if len(self.rolling_buys) >= self.WINDOW_SIZE:
            buy_window = self.last_window(self.rolling_buys)
            ask_window = self.last_window(self.rolling_asks)
            price = ((buy_window + ask_window)/2).mean()
            std = ((buy_window + ask_window)/2).std()
            acceptable_ask = price + self.Z_THRESH * std
            acceptable_bid = price - self.Z_THRESH * std
            orders, remaining_bids, remaining_asks, buy_pos, sell_pos = self.match_orders(acceptable_bid, acceptable_ask, order_depth, state.position.get(self.product, 0))
            
            if 0 > sell_pos > -20:
                orders.append(Order(self.product, int(min(ask_window)), 0 - sell_pos))
            elif 0 < buy_pos < 20:
                orders.append(Order(self.product, int(max(buy_window)), 0 - buy_pos))

            # StaticTrader.marketmake(product=self.product, tradeMade="BUY", acceptablePrice=(best_ask + best_bid) / 2, volume=20, orderList=orders)

            # StaticTrader.marketmake(product=self.product, tradeMade="SELL", acceptablePrice=(best_ask + best_bid) / 2, volume=20, orderList=orders)
        return orders

            



class Trader:

    def __init__(self) -> None:
        self.amTrader = MeanReversion(10, 0, "AMETHYSTS")
        self.sfTrader = MeanReversion(5, 0.3, "STARFRUIT")
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        print(state)

				# Orders to be placed on exchange matching engine
        # result = {}
        # for product in state.order_depths:
        #     order_depth: OrderDepth = state.order_depths[product]
        #     orders: List[Order] = []
        #     acceptable_price = 10  # Participant should calculate this value
        #     print("Acceptable price : " + str(acceptable_price))
        #     print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
        #     if len(order_depth.sell_orders) != 0:
        #         best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        #         if int(best_ask) < acceptable_price:
        #             print("BUY", str(-best_ask_amount) + "x", best_ask)
        #             orders.append(Order(product, best_ask, -best_ask_amount))
    
        #     if len(order_depth.buy_orders) != 0:
        #         best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        #         if int(best_bid) > acceptable_price:
        #             print("SELL", str(best_bid_amount) + "x", best_bid)
        #             orders.append(Order(product, best_bid, -best_bid_amount))
            
        #     result[product] = orders
        result = {}
        result['AMETHYSTS'] = self.amTrader.make_am_orders(state)
        result['STARFRUIT'] = self.sfTrader.make_sf_orders(state)
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        # logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData


def main():
    from datamodel import OrderDepth, TradingState, Order, Trade, Listing, Symbol
    timestamp = 1000

    listings = {
        "AMETHYSTS": Listing(
            symbol="AMETHYSTS", 
            product="AMETHYSTS", 
            denomination= "SEASHELLS"
        ),
        "STARFRUIT": Listing(
            symbol="STARFRUIT", 
            product="STARFRUIT", 
            denomination= "SEASHELLS"
        ),
        "COCONUTS": Listing(
            symbol="COCONUTS", 
            product="COCONUTS", 
            denomination= "SEASHELLS"
        ),
        "PINA_COLADAS": Listing(
            symbol="PINA_COLADAS", 
            product="PINA_COLADAS", 
            denomination= "SEASHELLS"
        ),
        "BERRIES": Listing(
            symbol="BERRIES", 
            product="BERRIES", 
            denomination= "SEASHELLS"
        ),
        "DIVING_GEAR": Listing(
            symbol="DIVING_GEAR", 
            product="DIVING_GEAR", 
            denomination= "SEASHELLS"
        ),
    }

    od = OrderDepth()
    od.buy_orders = {10: 7, 9: 5}
    od.sell_orders = {11: -4, 12: -8}

    od2 = OrderDepth()
    od2.buy_orders = {142: 3, 141: 5}
    od2.sell_orders = {144: -5, 145: -8}

    od3 = OrderDepth()
    od3.buy_orders = {142: 3, 141: 5}
    od3.sell_orders = {144: -5, 145: -8}

    od4 = OrderDepth()
    od4.buy_orders = {142: 3, 141: 5}
    od4.sell_orders = {144: -5, 145: -8}

    od5 = OrderDepth()
    od5.buy_orders = {142: 3, 141: 5}
    od5.sell_orders = {144: -5, 145: -8}


    od6 = OrderDepth()
    od6.buy_orders = {142: 3, 141: 5}
    od6.sell_orders = {144: -5, 145: -8}

    order_depths = {
        "AMETHYSTS": od,
        "STARFRUIT": od2,	
        "COCONUTS": od3,
        "PINA_COLADAS": od4,	
        "BERRIES": od3,
        "DIVING_GEAR": od4,	
    }

    own_trades = {
        "AMETHYSTS": [],
        "STARFRUIT": [],
        "COCONUTS": [],
        "PINA_COLADAS": [],
        "BERRIES": [],
        "DIVING_GEAR": [],	

    }

    market_trades = {
        "AMETHYSTS": [
            Trade(
                symbol="AMETHYSTS",
                price=11,
                quantity=4,
                buyer="",
                seller="",
                timestamp=900
            )
        ],
        "STARFRUIT": []
    }

    position = {
        "AMETHYSTS": 3,
        "STARFRUIT": -5,
        "COCONUTS": 3,
        "PINA_COLADAS": -5
    }

    observations = {}

    state = TradingState(
        timestamp=timestamp,
        traderData="SAMPLE",
        listings=listings,
        order_depths=order_depths,
        own_trades = own_trades,
        market_trades = market_trades,
        position = position,
        observations = observations
    )
    
    trader1 = Trader()
    for i in range(300):
        trader1.run(state)

# trader1.plotSpread()

if __name__ == "__main__":
    main()