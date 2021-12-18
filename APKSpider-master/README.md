# APKSpider
Spider for Android application

# 原因
为了对大量Android样本进行分析，我们需要从不同的应用市场中对应用进行爬取


# 目标
使用Python实现对于小米商城中的APK进行爬取，实现自动化爬取功能。

# 过程
1.  解析对应网页元素，获取APK链接
2.  获取每一个APK链接中下载链接
3.  对下载链接进行访问，将返回数据保存到本地
4. 找到是否有下一页的元素，从来获取下一页页面地址，重复上述操作
访问下一页的操作十分必要，这样才能够实现整个系统的自动化

# 提高
下一步使用Celery来实现并发


# XiaomiCrawler.py
第一个版本：小米应用商城爬虫，单进程模式，效率不高

# XiaomiCrawlerWithCelery.py
第二个版本：使用Celery实现并发爬取