#!/usr/bin/env python3
# -*-coding:utf8 -*-
# @TIME     :2018/11/30 下午1:43
# @Author   :hwwu
# @File     :fcxd.py

import requests
import re
from bs4 import BeautifulSoup
import os
import shutil
import json
from urllib import parse

# https://2wk.com/vip.php?url=
# 这个网址能解析的视频都可以通过这个下载

headers = {
    # 'Access-Control-Allow-Credentials': 'true',
    # 'Cache-Control': 'max-age=900',
    # 'Content-Encoding': 'gzip',
    # 'Content-Language': 'zh-CN',
    # 'Content-Type': 'text/html; charset=UTF-8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
    # 'Upgrade-Insecure-Requests': '1'
}

y_url2 = 'http://www.iqiyi.com/lib/m_209926914.html?src=search'
y_target2 = requests.get(url=y_url2).text
y_soup2 = BeautifulSoup(y_target2, 'html.parser')
y_returnsoup2 = y_soup2.find_all('div', attrs={'class': 'site-piclist_pic'})

# 用正则表达式获取剧集链接
y_result2 = re.findall('(?<=href=\").*?(?=\")', str(y_returnsoup2))
# 用正则表达式获取剧集名称
title2 = re.findall('(?<=title=\").*?(?=\">)', str(y_returnsoup2))
j = len(title2)
# 输出爬取结果
for i in range(2, j - 2):
    str1 = '第' + str(i + 1) + '集'
    print(y_result2[i])
    print(str1, title2[i])
    xm_url = 'http://aikan-tv.com/?url=' + y_result2[i]
    req = requests.get(xm_url, headers=headers)

    soup1 = BeautifulSoup(req.text, 'html.parser')
    returnsoup1 = soup1.find_all('iframe')
    result1 = re.findall('(?<=src=\").*?(?=\")', str(returnsoup1))

    req = requests.get(result1[0], headers=headers)
    req_json = re.findall('"api1.php", (.+),', req.text)[0]
    info = json.loads(req_json)

    data = {'time': info['time'], 'key': info['key'],
            'url': info['url'], 'type': info['type'], 'referer': info['referer']}
    req = requests.post('https://yun.odflv.com/odflv2/api1.php', headers=headers, data=data)
    info = json.loads(req.text)
    url = info['url']

    url1 = parse.unquote(url)

    req = requests.get(url1, headers=headers)
    result2 = re.findall('(?<=/).*?(?=.m3u8)', str(req.text))

    req = requests.get('https://acfun.iqiyi-kuyun.com/' + result2[0] + '.m3u8', headers=headers)
    text = req.text
    tl = text.split('\n')
    new_index = []
    for l in tl:
        if l.find('.ts') > 0:
            new_index.append(l)
    print(len(new_index), new_index)
    file_path = '/Users/liyangyang/Downloads/pachong/iqiyi/mid/fcxd/' + str1
    os.makedirs(file_path, exist_ok=True)
    for ii, ni in enumerate(new_index):
        url = 'https://acfun.iqiyi-kuyun.com' + ni
        r = requests.get(url, headers=headers)
        # print(url)
        content_length = int(r.headers['Content-Length'])
        path = file_path + '/' + str(ii) + '.ts'
        with open(path, 'ab') as file:
            file.write(r.content)
            file.flush()
            print(ni, 'receive data，file size : %d' % (content_length))

    new_path = '/Users/liyangyang/Downloads/pachong/iqiyi/result/fcxd/' + str1
    os.makedirs(new_path, exist_ok=True)
    exec_str = "cat " + file_path + '/*.ts  > ' + new_path + '/' + title2[i] + '.ts'
    print(exec_str)
    os.system(exec_str)
    shutil.rmtree(file_path)
    # sec_str = 'ffmpeg -y -i ' + new_path + '/new.ts -c:v libx264 -c:a copy -bsf:a aac_adtstoasc ' + new_path + '/new.mp4'
    # print(sec_str)
    # os.system(sec_str)
