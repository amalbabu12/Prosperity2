from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math
import numpy as np
import operator

class StaticTrader:
    limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}

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
                print(trade_made, str(vol_to_trade) + "x", prices)
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

            print(trade_made, str(vol_to_trade) + "x", prices)
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

    # def make_orders(self, state):
    #     orders = []
    #     order_depth = state.order_depths[self.product]
    #     position = state.position.get(self.product, 0)
    #     best_ask, _ = list(order_depth.sell_orders.items())[0]
    #     best_bid, _ = list(order_depth.buy_orders.items())[0]
    #     self.rolling_buys.append(best_ask)
    #     self.rolling_asks.append(best_bid)
    #     if len(self.rolling_buys) >= self.WINDOW_SIZE:
    #         buy_window = self.last_window(self.rolling_buys)
    #         ask_window = self.last_window(self.rolling_asks)
    #         price = (buy_window + ask_window).mean()//2

    #         next_ask = None
    #         next_buy = None
    #         for ask, quantity in list(order_depth.sell_orders.items()):
    #             next_ask = ask
    #             if ask < price:
    #                 position -= quantity
    #                 orders.append(Order(self.product, ask, -quantity))
    #             else:
    #                 break
    #         for bid, quantity in list(order_depth.buy_orders.items()):
    #             next_buy = bid
    #             if bid > price:
    #                 position -= quantity
    #                 orders.append(Order(self.product, bid, -quantity))
    #             else:
    #                 break

    #         if next_ask and next_ask - 1 > price and position > 0:
    #             orders.append(Order(self.product, next_ask - 1, -min(position, 20)))
    #         if next_buy and next_buy + 1 < price and position < 0:
    #             orders.append(Order(self.product, next_buy + 1, -max(position, 20)))
    #     return {self.product : orders}


    # - return the list of orders, (AMETHYST)
    # acceptable price - 10000, spread - 2
    # < 10k for working acceptable bids
    # > 10k for working acceptable asks
    def make_am_orders(self, state):
        orders = []
        order_depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]
        self.rolling_buys.append(best_ask)
        self.rolling_asks.append(best_bid)
        if len(self.rolling_buys) >= self.WINDOW_SIZE:
            # buy_window = self.last_window(self.rolling_buys)
            # ask_window = self.last_window(self.rolling_asks)
            price = 10000
            for ask, quantity in list(order_depth.sell_orders.items()):
                if ask < price:
                    position -= quantity
                    orders.append(Order(self.product, ask, -quantity))
                else:
                    break
            for bid, quantity in list(order_depth.buy_orders.items()):
                if bid > price:
                    position -= quantity
                    orders.append(Order(self.product, bid, -quantity))
                else:
                    break
        return orders
    

    # STARFRUIT
    # acceptable price - , spread -
    def make_sf_orders(self, state):
        orders = []
        order_depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]
        self.rolling_buys.append(best_ask)
        self.rolling_asks.append(best_bid)
        if len(self.rolling_buys) >= self.WINDOW_SIZE:
            buy_window = self.last_window(self.rolling_buys)
            ask_window = self.last_window(self.rolling_asks)
            price = ((buy_window + ask_window)/2).mean()
            std = ((buy_window + ask_window)/2).std()


            for ask, quantity in list(order_depth.sell_orders.items()):
                if (ask - price) / std < -self.Z_THRESH:
                    position -= quantity
                    orders.append(Order(self.product, ask, -quantity))
                else:
                    break
            for bid, quantity in list(order_depth.buy_orders.items()):
                if (bid - price) / std > self.Z_THRESH:
                    position -= quantity
                    orders.append(Order(self.product, bid, -quantity))
                else:
                    break
        return orders

            



class Trader:

    def __init__(self) -> None:
        self.amTrader = MeanReversion(10, 1, "AMETHYSTS")
        self.sfTrader = MeanReversion(10, 0.5, "STARFRUIT")
    
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