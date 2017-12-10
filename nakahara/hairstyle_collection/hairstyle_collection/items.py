# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class HairstyleCollectionItem(scrapy.Item):
    """Webページ内の画像の取得
    """

    image_urls = scrapy.Field()
    images = scrapy.Field()
