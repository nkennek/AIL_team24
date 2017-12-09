#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import os

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from hairstyle_collection.items import HairstyleCollectionItem



class HairStyleSpider(CrawlSpider):
    name = 'HairStyleSpider'
    allowed_domains = ['beauty.hotpepper.jp']
    start_urls = ["https://beauty.hotpepper.jp/catalog/ladys"]

    list_allow = [
        r'/catalog/.*',
        r'/sln.*',
        r'/trend.*',
        r'/doc.*',
        r'/CSP.*'
    ]
    list_deny = [
        r'/catalog/mens.*'
    ]
    rules = (
        #巡回ルール
        Rule(
            LinkExtractor(allow=list_allow, deny=list_deny, unique=True),
            callback='parse_imgs'
        ),
        Rule(
            LinkExtractor(allow=list_allow, deny=list_deny, unique=True),
            follow=True
        ),
    )

    hairstyle_url_re = r'https://imgbp\.hotp\.jp/CSP/IMG_SRC/.*\.(jpg|png)'

    def parse_imgs(self, response):
        hxs = scrapy.Selector(response)

        item = HairstyleCollectionItem()
        image_urls = hxs.xpath('//img/@src').extract()
        item['image_urls'] = self.parse_img_urls(image_urls)
        yield item

    def parse_img_urls(self, urls):
        """
        urlのパース

        - ドメイン名とパスで人物の写っている写真に限定
        - サイズ指定のないurlに変換
        """

        def sanitize(url):
            base_path, filename = os.path.dirname(url), os.path.basename(url)
            filename_base, ext = os.path.splitext(filename)
            if '_' in filename_base:
                filename_base = filename_base.split('_')[0]

            filename = filename_base + ext

            return os.path.join(base_path, filename)


        hairstyle_images = list(filter(lambda url: re.match( self.hairstyle_url_re, url), urls))
        return list(map(sanitize, hairstyle_images))
