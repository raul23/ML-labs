def print_(msg, is_data=1, newlines_for_nondata=False):
    if is_data:
        print(msg, end="\n\n")
    else:
        newlines = "\n\n" if newlines_for_nondata else "\n"
        print("*** {} ***".format(msg), end=newlines)
