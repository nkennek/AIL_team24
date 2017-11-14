#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from hairstyle_collection.items import HairstyleCollectionItem



class HairStyleSpider(CrawlSpider):
    name = 'HairStyleSpider'
    allowed_domains = ['beauty.hotpepper.jp']
    start_urls = ["https://beauty.hotpepper.jp/svcSA/"]

    list_allow = [
        r'/catalog/.*',
        r'/sln.*',
        r'/trend.*',
        r'/doc.*',
    ]
    list_deny = [
        r'/CSP.*'
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

    hairstyle_urls = r'https://imgbp\.hotp\.jp/CSP/IMG_SRC/.*\.(jpg|png)'

    def parse_imgs(self, response):
        hxs = scrapy.Selector(response)

        item = HairstyleCollectionItem()
        image_urls = hxs.xpath('//img/@src').extract()
        item['image_urls'] = list(filter(lambda url: re.match( self.hairstyle_urls, url), image_urls))
        yield item
