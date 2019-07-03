#! /usr/bin/env python
# -*- coding:utf-8 -*-
from plan import Planner
from generate import Generator
from random import randint



if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        keyword = input('Input Keyword:\t').strip()
        if keyword.lower() == 'quit' or keyword.lower() == 'exit':
            break
        gender = input('Input Gerder(1 for male, 0 for female):\t')
        try:
            if int(gender):
                g = u'男'
            else:
                g = u'女'
        except:
            g = u'女'
        surname = input('Input Surname:\t')
        n = input('How Many names:\t')
        try:
            n = int(n)
        except:
            n = 1
        if len(keyword) > 0:
            print('Keyword:', keyword, 'Gender:', g, 'Surname:', surname)
            print('\n')
            for _ in range(n):
                keywords = planner.plan(keyword + ' '+ g + ' ' + surname)

                sentences = generator.generate_name(keywords[0], keywords[2], keywords[1])
                print("Poem Generated:")
                print('\t' + sentences[0] + u'，' + sentences[1] + u'。')
                print("Name Generated:")
                f = randint(0, len(sentences[2]) - 1)
                s = randint(0, len(sentences[2]) - 1)
                while s == f:
                    s = randint(0, len(sentences[2]) - 1)
                print('\t' + surname + sentences[2][f] + sentences[2][s])
                print()

