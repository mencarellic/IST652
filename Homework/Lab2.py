
stock = {"banana": 6, "apple": 0, "orange": 32, "pear": 15}
prices = {"banana": 4, "apple": 2, "orange": 1.5, "pear": 3}
groceries = ['apple', 'banana', 'pear']

## Part a
print('------- Part a -------')
print('Stock of Oranges: {}'.format(stock['orange']))
stock['cherry'] = 14
prices['cherry'] = 2.4

## Part b
print('\n------- Part b -------')
for fruit in stock:
    print('{}: {}'.format(fruit, stock[fruit]))

## Part c
print('\n------- Part c -------')
instock = 0
for fruit in stock:
    if fruit in groceries:
        instock += stock[fruit]
print('Stock of groceries on list: {}'.format(instock))

## Part d
print('\n------- Part d -------')
for fruit in stock:
    print('{}: ${:.2f}'.format(fruit, float(prices[fruit] * stock[fruit])))


##### OUTPUT #####
# ------- Part a -------
# Stock of Oranges: 32
#
# ------- Part b -------
# banana: 6
# apple: 0
# orange: 32
# pear: 15
# cherry: 14
#
# ------- Part c -------
# Stock of groceries on list: 21
#
# ------- Part d -------
# banana: $24.00
# apple: $0.00
# orange: $48.00
# pear: $45.00
# cherry: $33.60