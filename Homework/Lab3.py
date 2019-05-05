## Problem 1
## This will print out d


## Problem 2
def computerpay(hours, rate):
    if hours > 40:
        overtime = hours - 40
        hours = 40
    else:
        overtime = 0

    pay = (hours * rate) + (overtime * (rate * 1.5))
    print('Gross pay: {:.2f}'.format(pay))

print('Enter hours and pay rate. Type done to exit')
while True:
    hours_input = input('Hours: ')
    rate_input = input('Rate: ')

    if hours_input.lower() == 'done':
        exit(1)
        break
    else:
        try:
            hours = float(hours_input)
        except Exception as e:
            print(e)
            exit(1)

    if rate_input.lower() == 'done':
        exit(1)
        break
    else:
        try:
            rate = float(rate_input)
        except Exception as e:
            print(e)
            exit(1)

    computerpay(hours, rate)


##### OUTPUT #####
# Enter hours and pay rate. Type done to exit
# Hours: 1
# Rate: 1
# Gross pay: 1.00
# Hours: 10
# Rate: 10
# Gross pay: 100.00
# Hours: 45
# Rate: 10
# Gross pay: 475.00
# Hours: 75
# Rate: 60
# Gross pay: 5550.00
# Hours: done
# Rate: done