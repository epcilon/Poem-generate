#! /usr/bin/env python
# -*- coding:utf-8 -*-

from generate import Generator
import sys




if __name__ == '__main__':
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
                sentences = generator.generate_name(keyword, g, surname)
                print("Poem Generated:")
                print('\t' + sentences[0] + u'，' + sentences[1] + u'。')
                print("Name Generated:")
                print('\t' + surname + sentences[2][:2])
                print()

