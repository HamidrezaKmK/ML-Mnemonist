exception_list = (AttributeError, FileNotFoundError)

if __name__ == '__main__':
    n = int(input())

    try:
        if n == 0:
            raise AttributeError
        elif n == 1:
            raise FileNotFoundError
        elif n == 2:
            raise Exception
        else:
            print("HOWDY!")
    except exception_list as e:
        print("Exception caught!")