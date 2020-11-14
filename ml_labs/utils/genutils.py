def print_(msg, is_data=1):
    if is_data:
        print(msg, end="\n\n")
    else:
        print("*** {} ***".format(msg))
