import os

def read_header_line(inputFile):
    fh = open(inputFile)
    header_list=fh.readline().split(" ")
    header_list=list(filter(('').__ne__, header_list))
    return header_list
