#-*- coding:utf-8 -*-

import urllib2
from lxml import etree
import os
import traceback

class xiaomiSpider():

    name = 'xiaomi'
    baseul = 'http://app.mi.com'
    downloadAPKPath = '/Users/qiuyu/Documents/apksamples/Xiaomi'  # 此处为下载地址，需要自己修改

    def download(self, url):
        response = urllib2.urlopen(url)
        tree = etree.HTML(response.read())
        nextPage = tree.xpath('//a[@class="download"]/@href')
        print(self.baseul + nextPage[0])
        apkName = url[29:] + '.apk'
        print(apkName)
        # 调用downloadApk
        self.downloadApk(self.baseul + nextPage[0], apkName)

    # 直接传递地址，下载文件，
    def downloadApk(self, url, apkName):

        if not os.path.exists(self.downloadAPKPath):
            os.makedirs(self.downloadAPKPath)
        apkpath = self.downloadAPKPath + '//' + apkName
        if os.path.exists(apkpath):
            print('the ' + apkName + '已经下载过') # 判断当前应用是否已经下载完成
            return
        else:
            try:
                f = urllib2.urlopen(url, timeout=600)
                print('开始下载...the url is:', url)
                data = f.read()
                with open(apkpath, 'wb') as code:
                    code.write(data)
                    f.close();
                    code.close();
                print('the ' + apkName + '已经下载完')
            except:
                traceback.print_exc()
                return

    def getApkSize(self, url):
        response = urllib2.urlopen(url)
        tree = etree.HTML(response.read())
        size = tree.xpath('/html/body/div[4]/div[1]/div[2]/div[2]/div/ul[1]/li[2]/text()')
        return size[0][:-2]

    def parseURL(self, url):
        response = urllib2.urlopen(url)
        tree = etree.HTML(response.read())
        count = 0;
        for sel in tree.xpath('/html/body/div[4]/div/div[1]/div[1]/ul/li'):
            # 取出每一个APK对应的URL
            print self.baseul + sel.xpath('h5/a/@href')[0]

            # 获取应用大小（由于这边只是做一个基本的演示，所以选取小一些的应用进行爬去，真实使用的时候可以在此处对应用大小进行限制）
            apkSize = self.getApkSize(self.baseul + sel.xpath('h5/a/@href')[0])
            if(float(apkSize) < 25):
                # 取出的每一个URL都调用download方法进行下载
                self.download(self.baseul + sel.xpath('h5/a/@href')[0])
                count +=1;
                if(count == 5):
                    break;
        nextPage = tree.xpath('//a[@class="next"]/@href')
        # 判断是否有下一页
        if nextPage:
            newxiayiye = url+nextPage[0]
            print(newxiayiye)
            # self.parseURL(newxiayiye) # 继续解析下一页的内容
        else:
            pass

if __name__ == '__main__':
    url = 'http://app.mi.com/topList'
    xiaomiSpider().parseURL(url)
