#!/usr/bin/env python
# encoding: utf-8

from PIL import Image
import sys
import os.path


EMPTY		= (255, 255, 255)
EMPTY_ASCII	= ' '
OBSTACLE	= (0, 0, 0)
OBSTACLE_ASCII	= 'X'
FOOD		= (255, 255, 0)
FOOD_ASCII	= 'f'
SPAWN		= (255, 0, 0)
SPAWN_ASCII	= 'o'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    img = Image.open(sys.argv[1])
    w, h = img.size
    print "Image size: {}×{}".format(w, h)
    print "Image mode: {}".format(img.mode)
    data = list(img.getdata())
    
    n_obstacles = 0
    n_food = 0
    n_spawn = 0
    
    ascii_matrix = list()
    for y in range(h):
        ascii_row = str()
        for x in range(w):
            pix = data[w*y + x]
            if pix == EMPTY:
                ascii_row += EMPTY_ASCII
            elif pix == FOOD:
                ascii_row += FOOD_ASCII
                n_food += 1
            elif pix == SPAWN:
                ascii_row += SPAWN_ASCII
                n_spawn += 1
            else:
                ascii_row += OBSTACLE_ASCII
                n_obstacles += 1
        ascii_row += '\n'
        ascii_matrix.append(ascii_row)
    
    filename = os.path.splitext(sys.argv[1])[0] + '.lvl'
    with open(filename, 'w') as f:
        f.writelines(ascii_matrix)
    print "{} converted to {}".format(sys.argv[1], filename)
    print "{} spawn location(s)".format(n_spawn)
    print "{} food entities".format(n_food)
    print "{}% obstacles".format(float(n_obstacles)*100/len(data))
