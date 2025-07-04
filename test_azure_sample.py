# %%
import os
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
def create_text_translation_client_with_credential():
    apikey = os.environ["AZURE_TEXT_TRANSLATION_KEY"]
    region = os.environ["AZURE_TEXT_TRANSLATION_REGION"]
    # [START create_text_translation_client_with_credential]
    credential = AzureKeyCredential(apikey)
    text_translator = TextTranslationClient(credential=credential, region=region)
    # [END create_text_translation_client_with_credential]
    return text_translator

# %%
from azure.core.exceptions import HttpResponseError

# -------------------------------------------------------------------------
# Text translation client
# -------------------------------------------------------------------------

text_translator = create_text_translation_client_with_credential()

# %%
def get_text_sentence_boundaries():
    # [START get_text_sentence_boundaries]
    try:
        from_language = "zh-Hans"
        source_script = "Latn"
        input_text_elements = [zhè shì gè cè shì。]

        response = text_translator.find_sentence_boundaries(
            body=input_text_elements, language=from_language, script=source_script
        )
        sentence_boundaries = response[0] if response else None

        if sentence_boundaries:
            detected_language = sentence_boundaries.detected_language
            if detected_language:
                print(
                    fDetected languages of the input text: {detected_language.language} with score: {detected_language.score}."
                )
            print(f"The detected sentence boundaries:")
            for boundary in sentence_boundaries.sent_len:
                print(boundary)

    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"Message: {exception.error.message}")
        raise
    # [END get_text_sentence_boundaries]

# %%
def get_text_sentence_boundaries_auto():
    # [START get_text_sentence_boundaries_auto]
    try:
        input_text_elements = ["This is a test. This is the second sentence."]

        response = text_translator.find_sentence_boundaries(body=input_text_elements)
        sentence_boundaries = response[0] if response else None

        if sentence_boundaries:
            detected_language = sentence_boundaries.detected_language
            if detected_language:
                print(
                    f"Detected languages of the input text: {detected_language.language} with score: {detected_language.score}."
                )
            print(f"The detected sentence boundaries:")
            for boundary in sentence_boundaries.sent_len:
                print(boundary)

    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"Message: {exception.error.message}")
    # [END get_text_sentence_boundaries_auto]

# %%
def split_text_by_lengths(text, lengths):
    sentences = []
    start = 0
    for length in lengths:
        # Slice the string from the current start index to the next "start+length"
        sentences.append(text[start:start+length])
        start += length
    return sentences

# Your original Chinese text (exactly as sent to the API)
chinese_text = (
    "Baker&Spice是上海Wagas（沃歌斯，后文为方便起见均用该名称代指Baker&Spice）旗下的餐厅。沃歌斯青岛首家餐厅位于万象城LG层中门外侧，意外不是很起眼的位置，在猪头肉印象里这家餐厅也相对比较低调，一直不温不火。正如其名，该餐厅的主营的糕点和意面披萨等简餐都有非常不错的水准。近期经常来沃歌斯，所以就做一下较深度点评。\\n\\n 沃歌斯的装修陈设简洁明亮，偏向北欧范儿。而没做吊顶的裸顶和垂下来的白炽灯又带着一丝工业风。高端文艺范的环境已经成功拉升了餐厅的格调。\\n 在沃歌斯点过两次沙拉，一次凯撒沙拉一次鸡肉芒果沙拉。虽然猪头肉并不喜欢吃生菜多的沙拉，不过沃歌斯的沙拉味道确实非常不错。凯撒沙拉是经典名菜了。沃歌斯的凯撒沙拉默认是不加鸡肉，加鸡肉需要加18元。猪头肉就后悔当时没有加钱加鸡肉。简版的凯撒沙拉是鹌鹑蛋、肉脯碎配蔬菜，上面撒着类似面包糠的碎屑。沃歌斯的沙拉首先胜在食材质量，鹌鹑蛋和生菜不说，肉脯碎的味道就很正，大有双鱼牌的赶脚。另外凯撒沙拉的的酱汁也非常地道。鸡肉芒果沙拉是这里沙拉类的销售冠军。鸡肉烤的火候适中，肉质紧实。芒果切的非常大块，吃到确实很爽。以牛油果为首的各类蔬果配菜也非常丰富。特制的酸甜味酱汁配奶油的味道清新独特，非常开胃。\\n\\n 意面半两尝过三款，各有特色，总体感觉都非常不错。细面配三文鱼通过烹制过程中加入少量奶油很好的烘托了三文鱼浓郁的味道。而选择在食用过程中挤一点柠檬汁则可以中和油腻感和厚重感，增加一丝清新。白葡萄酒海鲜意面是意面类的销售冠军，辅料是虾仁、比管、花蛤还有一点圣女果干和西芹。这款意面比其他意面都要少油，相对的却可以明显的尝出白葡萄酒的清香，而圣女果干沉淀的酸甜和西芹的水嫩很好的配合了白葡萄酒的香味。使用的海鲜鲜度也非常高，虾仁出自本地鲜活的海捕虾。可以明显尝得出来整道意面并没有使用提鲜的调味料，鲜度完全来自于以海虾为首的海鲜。绿酱鸡肉意面使用的罗勒酱味道非常纯正，单是罗勒酱就很赞了。配上奶油口感浓醇爽滑，搭配的空心粉和鸡肉也超合拍。这道意面强力推荐。\\n\\n 沃歌斯还有几款东南亚风的咖喱饭，猪头肉点过其中一款泰式辣味牛肉。这款咖喱是由牛肉、泡发海菜、菜椒和腰果烹制的。牛肉选用的是类似里脊牛腩的部位，肉质很嫩又有嚼劲。咖喱的香辛料绘制出的鲜辣味道很过瘾，而海菜的谷氨酸做了出色的点缀。咖喱的汤汁比较浓稠，配菜的长粒香米质量也很好。配在一起整道菜的体验还是很不错的。\\n\\n 披萨试过两款，意式海鲜披萨和火腿蘑菇披萨。沃歌斯的披萨都是薄饼底。可能是在面团里放了小苏打，披萨的饼底非常蓬松酥脆。披萨的芝士也是超级多，味道也很正，跟特制的薄饼底正是绝配。意式海鲜披萨使用的海鲜与海鲜意面相同，也是突出了以海虾为首的海鲜带来的鲜味。而火腿蘑菇披萨则以口菇特有的鲜味很好的衬托了优质火腿的肉香。这两款披萨都非常推荐。\\n\\n 沃歌斯的甜品品质也非常不错。树莓巧克力挞的酱汁酸味很重，是纯正的树莓果酱。芒果杏仁芝士蛋糕的芒果切得也很大块，中间夹的应该是奶油和芝士混合的酱，饼底的杏仁蛋糕味道很正。胡萝卜蛋糕的酱类似卡仕达酱，但很甜，就像混合了蜂蜜或者炼乳，口感非常醇厚。胡萝卜蛋糕的饼底口感也是醇厚到类似粗粮蛋糕或是枣糕的感觉，里面有胡萝卜和山核桃，也是非常不错。\\n\\n 沃歌斯的简餐和甜品水平都比较高，在同等价位里算是制作非常精致味道比较考究的了。面包没尝过，不妄作评断。目前不是很火爆，所以环境还是很好的。缺点是临街店内会有飞虫。另外猪头肉爆料一下，楼上CGV影院的会员卡在这里除了饮品之外都可享七五折，在万象城性价比还是蛮高的。"
)
english_text = (
    "Baker &Spice is a restaurant owned by Wagas in Shanghai. The first restaurant in Qingdao is located outside the middle gate of the LG floor of the MixC, which is unexpectedly not very inconspicuous, and this restaurant is relatively low-key in the impression of pork head meat, and has been tepid. As the name suggests, the restaurant's main menu of pastries, pasta, pizza and other light meals is of a very good standard. I've been coming to Wagas a lot lately, so I'll make a more in-depth review. \n\nWagas's décor is simple and bright, with a Nordic twist. And the bare ceiling without a suspended ceiling and the hanging incandescent lamp have a hint of industrial style. The high-end literary environment has successfully elevated the restaurant's style. \n Ordered two salads at Wagas, one Caesar salad and one chicken mango salad. Although pork head doesn't like lettuce-heavy salads, Wagas salads do taste very good. Caesar salad is a classic dish. Wagas's Caesar salad is no chicken by default, and it costs 18 yuan to add chicken. The pig's head regretted that it didn't add money and chicken at that time. The short version of Caesar salad consists of quail eggs, minced meat and vegetables, sprinkled with breadcrumb-like crumbs. Wagas's salad is first and foremost about the quality of the ingredients, not to mention the quail eggs and lettuce, and the taste of the minced meat is very positive. In addition, the dressing of the Caesar salad is very authentic. Chicken and mango salad is the sales champion of the salad category here. The chicken is grilled at a moderate temperature and the meat is firm. The mango is cut into very large pieces, and it is really refreshing to eat. There are also a variety of side dishes such as avocados. The special sweet and sour sauce with cream has a fresh and unique flavor and is very appetizing. \n\n I have tried three types of pasta in half, each with its own characteristics, and the overall feeling is very good. The thin noodles with salmon are cooked with a small amount of cream to bring out the rich flavor of the salmon. Choosing to squeeze a little lemon juice during consumption can neutralize the oiliness and heaviness and add a hint of freshness. White wine seafood pasta is the top seller in the pasta category, with shrimp, biguan, clams and a dash of dried cherry tomatoes and celery. This pasta is less oily than other pastas, but you can clearly taste the aroma of white wine, and the sweetness and sourness of dried cherry tomatoes and the tenderness of celery are well matched by the aroma of white wine. The freshness of the seafood used is also very high, and the shrimp are made from local fresh sea-caught shrimp. It is evident that the whole pasta dish does not use seasonings to enhance the freshness, and the freshness comes entirely from the seafood including shrimp. The basil sauce used in the green sauce chicken pasta has a very pure taste, and the basil sauce alone is amazing. It is served with cream, and the hollow flour and chicken go well with it. This pasta dish is highly recommended. Wagas also has a couple of Southeast Asian-inspired curries and rice, with pork head and one of the Thai spicy beef. This curry is made with beef, pickled seaweed, peppers and cashew nuts. The beef is made with a portion similar to a tenderloin brisket, and the meat is tender and chewy. The spices of the curry create a spicy flavor that is enjoyable, and the glutamate of the sea vegetables is an excellent garnish. The soup of the curry is thick, and the long-grain basmati rice of the side dishes is of good quality. The experience of the whole dish together is still very good. \n\nI tried two types of pizza, the seafood pizza and the ham and mushroom pizza. Wagas pizzas are all pizza bases. Probably with baking soda in the dough, the pizza base is very fluffy and crispy. The pizza has a lot of cheese and tastes good, and it goes well with the special pizza base. The seafood pizza uses the same seafood as the seafood pasta, but it also highlights the umami flavor of seafood such as shrimp. The ham mushroom pizza sets off the meaty aroma of high-quality ham with the unique umami flavor of mushrooms. Both pizzas are highly recommended. The quality of Wagas' desserts is also very good. The sauce of the raspberry chocolate tart has a strong sour flavor and is pure raspberry jam. The mango almond cheesecake is also cut into large pieces, and the sauce mixed with cream and cheese is sandwiched in the middle, and the almond cake at the bottom of the cake tastes very good. The carrot cake sauce is similar to custard sauce, but it is sweet, like a mixture of honey or condensed milk, and has a very mellow taste. The texture of the carrot cake is also mellow to the feeling of coarse grain cake or date cake, and it contains carrots and pecans, which is also very good. The light meals and desserts at Wagas are of a relatively high standard, and they are very delicate and tasteful at the same price. I have not tasted bread, and I do not judge. It's not very popular at the moment, so the environment is still very good. The disadvantage is that there are flying insects in the storefrontage. In addition, the pig's head meat broke the news that the membership card of the CGV theater upstairs can enjoy a 75% discount here in addition to drinks, and the cost performance in the MixC is quite high."
)

# The list of sentence lengths you mentioned
lengths = [56, 59, 32, 21, 25, 25, 21, 29, 34, 11, 26, 16, 32, 43, 16, 18, 14, 16, 19, 24, 29, 34, 35, 41, 63, 25, 40, 29, 25, 9, 37, 23, 25, 33, 23, 17, 26, 12, 25, 29, 38, 30, 11, 19, 23, 47, 37, 47, 44, 12, 18, 12, 50]
# lengths = [58, 241, 125, 77, 61, 116, 84, 78, 87, 32, 84, 76, 125, 168, 65, 74, 71, 77, 58, 98, 121, 113, 131, 141, 224, 113, 158, 115, 75, 39, 117, 72, 99, 130, 103, 62, 82, 34, 82, 93, 129, 111, 36, 50, 94, 194, 140, 157, 132, 45, 76, 71, 205]

# Use the helper function to split the text
sentences = split_text_by_lengths(english_text, lengths)

# Show each sentence with its character count for verification
for idx, source_sentence in enumerate(sentences, 1):
    print(f"Sentence {idx} ({len(source_sentence)} chars): {source_sentence}")

# %%
print(sentences)

# %%
import polars as pl

# %%
data = {"col1": [0, 2], "col2": [3, 7]}

df = pl.LazyFrame(data, schema={"col1": pl.Float32, "col2": pl.Int64})

# %%
def my_function(x):
    return x*2

# %%
df.select(pl.len()).collect().item()

# %%
batch_size = 1000
lazy_batches = [
    df.slice(i, batch_size).lazy() for i in range(0, df.select(pl.len()).collect().item(), batch_size)
]

transformed = [lazy.map_batches(my_function) for lazy in lazy_batches]
# result = pl.concat([lf.collect() for lf in transformed])

# %%
text = 'Zhuangyuanlou Hotel went for the first time, because of the geographical location: in Ningbo City and Yi Avenue high, big, on, inside the decoration of Chinese, the dish is authentic Ningbo cuisine, the taste is pure, drunk mud snail is particularly good, eat the taste of childhood, because I went late, waited in the lobby for a while, during which there is tea to drink, the waiter also chats with you, to the dining business is too good, the waiter is trotting, the service attitude is absolutely not speedy, everything is in place, the drink is also patiently explained to us, so it is absolutely necessary to boast, In particular, Peng Xinxing and Hong Jihua (only know the name by looking at the service card) also add color to our image of Ningbo, the champion building is a window in Ningbo, and the quality of the waiter reflects the spiritual outlook of our Ningbo people. Like one'
text[0:623]

# %%
import polars as pl

# %%
df = pl.read_parquet('./text-zh-simplified_translated.parquet')

# %%
df.head(5)

# %%
text = "I love their pork knuckles, spicy chicken feet, meat slices and mouth grinding, the dishes are all home-cooked taste, very delicious, it is the original Haoye, and the Qiangye restaurant in the north of Changbai Street, it is a sister shop, the same but not too the same, to make up for one of Qiangye's regrets, that is, Qiangye has no supper now, and there is a supper at my mother's house, and you can come over after 10 o'clock. The overall dining environment is clean and tidy, simple and refreshing, with two floors. The service is also in place! The owner is super nice and especially easy to talk to! Nanjing's authentic local cuisine needs to be supported!! It is suitable for a small gathering of relatives and friends, and the bus stop is at the door, which is very convenient!"
lengths = [
            434,
            90,
            30,
            56,
            57,
            121
        ]

# %%
def split_text_by_word_revised(text, lengths):
    sentences = []
    start = 0
    for i, length in enumerate(lengths):
        end = min(start + length, len(text))
        if i < len(lengths) - 1: # For all segments except the last
            while end > start and text[end-1] != ' ':
                end -= 1
            if end == start:
                end = min(start + length, len(text)) # Force split if no space found
        else:
            end = len(text) # For the last segment, just take the rest of the text
        sentences.append(text[start:end].strip())
        start = end
    return sentences

# %%
split_text_by_word_revised(text, lengths)

# %%
source_split_column= "source_split_texts"
translated_split_column= "translated_split_texts"
new_rows= []
for row in df.head(5).iter_rows(named=True):
    sentence_index = 0
    for source_sentence, translated_sentence in zip(row[source_split_column], row[translated_split_column]):
        if source_sentence.strip():
            new_rows.append({"id": row["id"], "sentence_index": sentence_index, "source_text": source_sentence, "translated_text": translated_sentence})
            sentence_index += 1
    # print(row[split_column])
        


# %%
pl.DataFrame(new_rows)

# %%
import polars as pl

# %%
pl.read_excel(source="JUL 2024- text comments.xlsx.xlsx", sheet_name="response_export_comments")

# %%
from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()
sentence = "Parlez-vous français? " + \
            "Ich spreche Französisch nur ein bisschen. " + \
            "A little bit is better than nothing." + \
            "---"
for result in detector.detect_multiple_languages_of(sentence):
    # print(f"{result.language.name}: '{sentence[result.start_index:result.end_index]}'")
    print(len(str(result.language.iso_code_639_1.name).lower()))



# %%
lang = detector.detect_language_of("-")
print(lang)

# %%
detected_languages = [result.language.iso_code_639_1.name.lower() if result.language.iso_code_639_1.name is not None else "unknown" for result in detector.detect_multiple_languages_of("-")]
print(detected_languages if detected_languages else "unknown")

# %%



