# Status report — Arabic and English reading engine

> **Written for:** the project owner and anyone they forward this to.
> **Date:** 2026-11-05 · build `18bbf43`
> **Bilingual page:** https://claude.ai/artifact/4FoV2XASkASAYxLYKWfTQK

Arabic first, English below. Both halves say the same thing.

Every number here was measured on this build. Nothing is estimated or
projected, and §7 lists what is still unproven.

---

# تقرير الحالة — محرّك قراءة النص العربي والإنجليزي

## ١. ما هو النظام

محرّك يقرأ نصًا **مطبوعًا** من صورة كاميرا — عربي وإنجليزي — ويُرجِع الكلمات
وأماكنها على الصفحة، **مع الحفاظ على التشكيل**: الفَتحة والضَّمّة والكَسرة
والشَّدّة والسُّكون.

التشكيل هو جوهر المنتج وليس تفصيلًا إضافيًا. الطفل الذي يتعلّم القراءة يحتاج
أن يعرف أن `كَتَبَ` تُقرأ غير `كُتِبَ` — نفس الحروف، نُطق مختلف، معنى مختلف.
أي محرّك يحذف العلامات يُنتج رقم دقة أجمل، ومنتجًا لا يستطيع أن يقول للطفل
كيف يَنطق الكلمة.

النظام يعمل **بالكامل على الجهاز**: لا خدمة سحابية، ولا رفع صور إلى أي جهة
خارجية، ولا تتبُّع. تم التحقق من ذلك عمليًا بتشغيل الاستدلال بعد حجب الاتصالات
الشبكية على مستوى `socket` — صفر محاولة اتصال.

## ٢. ما الذي بُني

- **خط المعالجة الكامل:** كشف مواضع النص، ثم التعرّف عليه، ثم ترتيب القراءة
  من اليمين إلى اليسار، ثم التطبيع (توحيد يونيكود، إزالة التطويل).
- **موديل عربي مُدرَّب خصيصًا** على نص مشكّل — انظر §٤.
- **قياس يفرِّق بين خطأ الحرف وخطأ العلامة:** دقة الحروف، دقة الكلمات، ثم
  استرجاع التشكيل *بحسب الموضع* — فعلامة صحيحة فوق الحرف الخطأ تُحسب خطأ،
  لأنها نُطق خطأ.
- **واجهة محلية** تعمل في المتصفح، مع نافذة كاميرا مقصوصة على مقاس A5 حتى
  تمتلئ الصورة بالصفحة وحدها، وخيار رفع صورة جاهزة.
- **مولِّد بيانات تدريب** يرسم سطورًا عربية مشكّلة بخطوط مطبعية حقيقية — ٢٥٠٠٠ سطر.
- **٤٥٥ اختبارًا آليًا** تُشغَّل في أقل من ٦ ثوانٍ، و١٥٢٠٠ سطر برمجي.

> **تمييز مهم:** الـ ٤٥٥ اختبارًا تُثبت أن **الكود** صحيح — المقاييس تحسب ما
> تقول إنها تحسبه، والتشكيل لا يُحذف في أي طبقة. لا تُثبت شيئًا عن **دقة
> القراءة**. هذان شيئان مختلفان، وخلطهما هو أسهل طريقة لتقديم رقم غير حقيقي.

## ٣. النتائج المقيسة

تسع صفحات فونكس وُلِّدت داخليًا، كل سطر فيها له نصّه الصحيح المعروف **بالضبط**،
فالنتيجة تُقاس ولا تُقدَّر بالنظر. شُغِّل خط المعالجة الكامل على كل صفحة على
كرت RTX 3050 Ti.

| الصفحة | الخط | دقة الحروف مع التشكيل | الحروف وحدها | استرجاع التشكيل | خطأ الكلمات |
|---|---|---|---|---|---|
| `ar_04_raa` | Sakkal Majalla | 95.9% | 98.3% | 0.925 (37/40) | 0.364 |
| `ar_01_baa` | Sakkal Majalla | 93.0% | 95.0% | 0.400 (16/40) | 0.182 |
| `ar_02_meem` | Sakkal Majalla | 92.2% | 90.9% | 0.388 (19/49) | 0.417 |
| `ar_03_seen` | Sakkal Majalla | 88.3% | 93.4% | 0.524 (22/42) | 0.615 |
| `ar_06_reading` | Sakkal Majalla | 87.7% | 86.2% | 0.415 (17/41) | 0.583 |
| `ar_07_naskh` | Traditional Arabic | 85.0% | 91.7% | 0.275 (11/40) | 0.636 |
| `ar_05_dal` | Sakkal Majalla | 80.2% | 79.0% | 0.359 (14/39) | 0.538 |
| `ar_08_naskh` | Amiri | 63.0% | 61.7% | 0.050 (2/40) | 1.182 |
| **العربية — ٨ صفحات** | — | **85.7%** | **87.0%** | **0.417 (138/331)** | **0.564** |
| `phonics_b_en` | Georgia | **100%** | 100% | — | **0.000** |

### كيف تُقرأ هذه الأرقام

- **الإنجليزية محلولة.** صفحة حرف B قُرئت بالكامل: ١٧ كلمة من ١٧، صفر خطأ.
- **العربية بدون تشكيل قوية؛ ومع التشكيل متوسطة.** ٨٥.٧٪ على مستوى الحرف تبدو
  مقبولة، لكن الرقم الذي يهمّ المنتج هو الكلمة: **٤٣.٦٪ فقط من الكلمات صحيحة
  بالكامل** (٥٣ خطأ في ٩٤ كلمة). الطفل يقرأ كلمات، لا حروفًا متفرقة.
- **التشكيل غير مستقرّ.** نفس الموديل، نفس الخط، صفحات متشابهة الصعوبة:
  استرجاع ٠.٩٢٥ في صفحة الراء و٠.٣٥٩ في صفحة الدال. التقلّب نفسه مشكلة،
  وليس المتوسط فقط.
- **أسوأ صفحة هي الأهم.** ٦٣.٠٪ على خط Amiri، وعلامتان من ٤٠. هذه الصفحة
  موضوعة في المجموعة *لتفشل* — §٥ يشرح لماذا.

> **تحذير صريح:** هذه صفحات **مُولَّدة رقميًا، وليست صورًا فوتوغرافية** لكتب
> حقيقية: إضاءة مثالية، تركيز مثالي، بلا ميل ولا انعكاس ولا ظل. **مُدخل
> الكاميرا الحقيقي أصعب في كل جانب.** هذه الأرقام تُثبت أن المحرّك يعمل من
> الطرف إلى الطرف وأن المقاييس سليمة. وليست نتيجة دقة للمنتج، و**هدف الـ ٩٥٪
> غير مُقيَّم بها.**

## ٤. التدريب الخاص: ما الذي تغيّر

الموديل الجاهز كان يقرأ الحروف ويُهمل التشكيل. دُرِّب على نص عربي مشكّل، وقيس
على صورة فوتوغرافية حقيقية لصفحة مطبوعة مع نصّها الصحيح المكتوب يدويًا:

| المقياس | الموديل الجاهز | بعد التدريب |
|---|---|---|
| دقة الحروف | 59.4% | **76.1%** |
| خطأ الكلمات | 0.864 | **0.773** |
| استرجاع التشكيل | 0.172 (11/64) | **0.453 (29/64)** |
| علامات مُختلَقة على نص غير مشكّل | 8 | **0** |

السطر الأخير مطلب صريح في مواصفة المنتج: على نص مطبوع **بلا** تشكيل، كان
الموديل يخترع ٨ علامات — والعلامة الخطأ نُطق خطأ. دورة تدريب ثانية أُضيف فيها
٣٠٪ من السطور بلا تشكيل أنزلت الرقم إلى **صفر** دون أن تخسر استرجاع التشكيل.

ملاحظة على الحدود: التدريب على العربية وحدها *دمّر* قدرة الموديل على
الإنجليزية (قرأ «The boy wrote the lesson» حروفًا عربية عشوائية)، فالإنجليزية
تستخدم موديلها المستقل. هذا قرار معماري مقصود، لا عجز.

## ٥. أهم اكتشاف: المشكلة في نوع الخط، لا في حجم الموديل

الموديل بعد التدريب قرأ صورة صفحة كتاب بنسبة خطأ ٠.٢٦٥، وحذف كل التشكيل من
الطرفين لم ينزل بها إلا إلى ٠.٢٠٩ — أي أن **الحروف نفسها** كانت خطأ، لا
العلامات وحدها. اختُبرت ثلاثة أسباب محتملة:

- **ليست قوة الكشف.** كاشف أكبر ١٨ مرة أعطى نفس نسبة الخطأ تمامًا (٠.٢٦٤٥)،
  وأسوأ في الكلمات والتشكيل. زيادة الحجم لا تُصلح شيئًا هنا.
- **ليست دقة الصورة.** ارتفاع السطر المقيس ١٣١ بكسل — أعلى بكثير من أي حدّ أدنى.
- **بل نوع الخط.** نفس الجملة رُسمت بثمانية خطوط وقُرئت: متوسط نسبة الخطأ
  **٠.٠٥٤ على خطوط الشاشة** مقابل **٠.٢٩٩ على خطوط النسخ** — أسوأ ٥.٥ مرة
  على النوع الوحيد الذي تُطبع به الكتب فعلًا.

السبب كان في مولِّد بيانات التدريب لا في الموديل: كان يوزّع الخطوط بالتساوي،
فنال Courier New و DejaVu Sans **Mono** — خط عربي ثابت العرض لا وجود له في
المطبوعات أصلًا — نفس عدد النماذج الذي نال «الخط التقليدي». صُحِّح التوزيع:
حُذفت ستة خطوط لا تُستخدم في الطباعة، ورُفعت حصة النسخ من ١٥٪ إلى ٦٦٪.

### وأثناء ذلك: المولِّد كان يرسم التشكيل في المكان الخطأ

عند بناء صفحات الاختبار ظهر خلل أعمق. الطريقة القديمة كانت تحوِّل الحروف إلى
«أشكال العرض» القديمة في يونيكود ثم ترسمها حرفًا حرفًا. هذه الطريقة **لا
تستطيع وضع حركة فوق حرف**: شكل العرض يحمل صورة الحرف المتّصل، لكن لا شيء يخبر
الراسم أن الفتحة تنتمي فوق الحرف الذي يسبقها — فيرسمها *بجانبه*.

من ثمانية خطوط، نجا خطان فقط. أما Amiri والخط التقليدي والأندلس وسكّال مجلة —
وهي خطوط النسخ المطبعية — فرسمت العلامات منفصلة وفي غير موضعها. أي أن **جزءًا
كبيرًا من بيانات التدريب كان خطأ طباعيًا**، وتعديل الأوزان نحو النسخ الذي
انتهى للتوّ كان *سيزيد الطين بلّة*، لأن خطوط النسخ هي بالضبط التي تُرسم خطأ.

**الحل:** أُضيفت وحدة رسم جديدة (`phonics_ocr.shaping`) تستخدم HarfBuzz و
FreeType، وهي ما يستخدمه نظام الطباعة الحقيقي: تقرأ جداول OpenType داخل الخط
نفسه، فتَحلّ الحركة حيث وضعها مصمّم الخط بالضبط. الثمانية خطوط كلها تُرسم الآن
صحيحة، ومنها خطان كان الفحص القديم مضطرًا لرفضهما.

ونتيجة لذلك: **دورة التدريب على خطوط النسخ أُوقفت عند الحلقة الأولى من ٢٠**،
لأنها كانت ستتدرّب على بيانات خطأ. تستحق أن تُعاد الآن بعد أن صار الرسم سليمًا،
وهذه أول خطوة فنية مقترحة.

## ٦. حقائق تشغيلية مُتحقَّق منها

| الحقيقة | الدليل |
|---|---|
| يعمل على كرت الرسوميات | تنفيذ فعلي على RTX 3050 Ti، CUDA 11.8 |
| يرتدّ إلى المعالج عند غياب الكرت | مُجرَّب؛ احتاج تعطيل oneDNN لتجاوز خلل في المكتبة |
| الكرت أسرع ٥ مرات | 0.686 s مقابل 3.449 s لنفس الصورة |
| يعمل بلا إنترنت | استدلال ناجح مع حجب الشبكة — صفر محاولة اتصال |
| حجم صغير يناسب جهازًا مدمجًا | ٢١ ميجابايت للموديلات الثلاثة |
| التشكيل موجود في قاموس الموديل | ٧٤٧ حرفًا، منها ٣٨ علامة عربية — الحركات الثماني كلها |
| صناديق على مستوى الكلمة | **غير متاحة من الكاشف** — يُرجِع سطورًا لا كلمات |

السطر الأخير له أثر على المنتج: إبراز كلمة بعينها على الصفحة يحتاج تقسيم السطر
إلى كلمات بخطوة مستقلة. هذه الخطوة مكتوبة ومُختبَرة، لكن مصدرها محاذاة مخارج
الموديل لا كشف مباشر للكلمات — وهو فرق يجب ألا يُقدَّم على أنه كشف حقيقي للكلمات.

## ٧. ما لم يُثبت بعد

- **هدف الـ ٩٥٪ غير مُقيَّم.** لا توجد بيانات اختبار حقيقية محجوزة. كل رقم في
  هذا التقرير إما من صفحات مُولَّدة أو من *صورة واحدة* موسومة يدويًا.
- **لا مجموعة اختبار فوتوغرافية.** المطلوب ٢٠–٥٠ صفحة مصوَّرة ومكتوبة نصوصها
  يدويًا قبل أن يكون لأي رقم دقة معنى إحصائي. هذه أعلى خطوة قيمةً في المشروع الآن.
- **دقة الكلمة العربية منخفضة** (٤٣.٦٪) حتى على مُدخل مثالي.
- **استرجاع التشكيل متقلّب** بين ٠.٠٥ و٠.٩٢ بلا سبب مفهوم بعد.
- **مسألة ترخيص.** مُدوَّنة التشكيل المستخدمة في التدريب مرخّصة GPL-2.0، وتحتاج
  مراجعة قانونية قبل الشحن داخل منتج مشمول ببراءة اختراع.

## ٨. الخطوات التالية بترتيب الأولوية

1. **تصوير وتوسيم ٢٠–٥٠ صفحة حقيقية.** بدونها لا يوجد رقم دقة يُقتبس. لا شيء
   في هذا التقرير يُعمَّم على مُدخل الكاميرا.
2. **إعادة توليد بيانات التدريب بالرسم الصحيح** ثم إعادة دورة التدريب على خطوط
   النسخ التي أُوقفت. هذا أقرب تحسُّن متوقَّع.
3. **قياس استرجاع التشكيل قياسًا سليمًا:** ٥٠ صفحة مشكّلة، موسومة مرتين بتشكيل
   وبدونه، حتى يُنسب الفرق إلى سببه.
4. **تحديد حدّ الدقة البصرية** الذي تصبح الفتحة عنده غير قابلة للقراءة. هذا يضع
   شرطًا صريحًا على عدسة الكاميرا في الجهاز.
5. **حسم مسألة تحديد الكلمة** (§٦)، لأن إبراز الكلمة على الصفحة يعتمد عليها.
6. **مراجعة ترخيص المُدوَّنة.**

---

# Status report — Arabic and English reading engine

## 1. What the system is

An engine that reads **printed** text from a camera image — Arabic and
English — and returns the words and where they sit on the page, **with
the diacritics preserved**: fatha, damma, kasra, shadda and sukun.

Tashkeel is the product, not a nicety. A child learning to read needs to
know that `كَتَبَ` and `كُتِبَ` are different words — the same letters, a
different pronunciation, a different meaning. An engine that drops the
marks produces a prettier accuracy figure and a product that cannot tell
a child how to say the word.

Everything runs **on the device**: no cloud service, no image upload to
any outside party, no telemetry. This was verified by running inference
with network connections blocked at the `socket` level — zero connection
attempts.

## 2. What has been built

- **The full pipeline:** text detection, recognition, right-to-left
  reading order, then normalisation (Unicode NFC, tatweel removal).
- **An Arabic recogniser fine-tuned** on vowelled text — see §4.
- **Metrics that separate a letter error from a mark error:** character
  accuracy, word accuracy, and diacritic recall scored *by position*, so
  a correct mark over the wrong letter counts as wrong — because it is
  the wrong pronunciation.
- **A local browser interface** with the camera cropped to A5, so the
  page fills the frame and nothing else does, plus an upload path for an
  existing image.
- **A training-data generator** that typesets vowelled Arabic in real
  print typefaces — 25,000 lines.
- **455 automated tests** running in under 6 seconds, over 15,200 lines
  of code.

> **An important distinction:** the 455 tests prove the **code** is
> correct — the metrics compute what they claim to compute, and no layer
> silently strips a diacritic. They prove nothing about **reading
> accuracy**. These are two different things, and conflating them is the
> easiest way to report a number that is not real.

## 3. Measured results

Nine phonics pages generated in-house, where every line has an
**exactly** known correct transcription, so the output is scored rather
than admired. The full pipeline ran on each page on an RTX 3050 Ti.

| Page | Typeface | Char. accuracy, with tashkeel | Letters only | Diacritic recall | Word error |
|---|---|---|---|---|---|
| `ar_04_raa` | Sakkal Majalla | 95.9% | 98.3% | 0.925 (37/40) | 0.364 |
| `ar_01_baa` | Sakkal Majalla | 93.0% | 95.0% | 0.400 (16/40) | 0.182 |
| `ar_02_meem` | Sakkal Majalla | 92.2% | 90.9% | 0.388 (19/49) | 0.417 |
| `ar_03_seen` | Sakkal Majalla | 88.3% | 93.4% | 0.524 (22/42) | 0.615 |
| `ar_06_reading` | Sakkal Majalla | 87.7% | 86.2% | 0.415 (17/41) | 0.583 |
| `ar_07_naskh` | Traditional Arabic | 85.0% | 91.7% | 0.275 (11/40) | 0.636 |
| `ar_05_dal` | Sakkal Majalla | 80.2% | 79.0% | 0.359 (14/39) | 0.538 |
| `ar_08_naskh` | Amiri | 63.0% | 61.7% | 0.050 (2/40) | 1.182 |
| **Arabic, 8 pages** | — | **85.7%** | **87.0%** | **0.417 (138/331)** | **0.564** |
| `phonics_b_en` | Georgia | **100%** | 100% | — | **0.000** |

### How to read these numbers

- **English is solved.** The letter-B page came back complete: 17 words
  of 17, no errors.
- **Unvowelled Arabic is strong; vowelled Arabic is middling.** 85.7% at
  the character level looks acceptable, but the figure the product lives
  on is the word: **only 43.6% of words are exactly right** (53 errors in
  94 words). A child reads words, not loose letters.
- **Diacritics are unstable.** Same model, same typeface, pages of
  comparable difficulty: recall 0.925 on the raa page and 0.359 on the
  dal page. The variance is itself a problem, not just the mean.
- **The worst page matters most.** 63.0% in Amiri, with 2 marks of 40
  found. That page is in the set *to fail* — §5 explains why.

> **Stated plainly:** these are **rendered pages, not photographs** of
> real books: perfect lighting, perfect focus, no skew, no glare, no
> shadow. **Real camera input is harder in every respect.** These numbers
> show the engine runs end to end and that the metrics behave correctly.
> They are not an accuracy result for the product, and **the 95% target
> is not evaluated by them.**

## 4. The fine-tune: what changed

The stock model read the letters and ignored the marks. It was fine-tuned
on vowelled Arabic and scored against a real photograph of a printed page
with a hand-written transcription:

| Metric | Stock model | After fine-tuning |
|---|---|---|
| Character accuracy | 59.4% | **76.1%** |
| Word error rate | 0.864 | **0.773** |
| Diacritic recall | 0.172 (11/64) | **0.453 (29/64)** |
| Invented marks on unvowelled print | 8 | **0** |

That last row is an explicit product requirement. On print carrying
**no** tashkeel, the model was inventing 8 marks — and a wrong mark is a
wrong pronunciation. A second training run, with 30% of its lines
carrying no marks at all, brought that to **zero** without losing
diacritic recall.

One limit worth stating: training on Arabic alone *destroyed* the model's
English — it read "The boy wrote the lesson" as random Arabic letters —
so English keeps its own separate model. That is a deliberate
architectural decision, not a shortcoming.

## 5. The main finding: the typeface, not the model size

The fine-tuned model read a photographed book page at a 0.265 error rate,
and stripping every diacritic from both sides only brought that to 0.209 —
so the **letters themselves** were wrong, not only the marks. Three
candidate causes were tested:

- **Not detection capacity.** A detector 18x larger gave an identical
  error rate (0.2645) and was worse on words and on diacritics. Scaling
  up fixes nothing here.
- **Not resolution.** Measured line height was 131 px, far above any floor.
- **The typeface.** One sentence rendered in eight faces and read back:
  mean error **0.054 on screen faces** against **0.299 on Naskh faces** —
  5.5x worse on the only kind of type printed books actually use.

The cause was in the training-data generator, not the model: it sampled
its typefaces uniformly, so Courier New and DejaVu Sans **Mono** —
monospaced Arabic, which does not occur in print at all — each got about
as many samples as Traditional Arabic. The distribution was corrected: six
faces print never uses were dropped, and Naskh was raised from a 15% share
of samples to 66%.

### And along the way: the generator was drawing the marks in the wrong place

Building the test pages exposed something deeper. The old path converted
letters into Unicode's legacy presentation forms and drew them one
codepoint at a time. That approach **cannot place a harakah at all**: a
presentation form carries the letter's joined shape, but nothing tells the
renderer that a fatha belongs centred above the letter it follows — so it
gets drawn *beside* it.

Of eight typefaces, two survived. Amiri, Traditional Arabic, Andalus and
Sakkal Majalla — the print Naskh faces — all drew the marks detached and
misplaced. Which means **much of the synthetic training set was
typographically wrong**, and the Naskh reweighting just completed would
have made things *worse*, since those are exactly the faces that
misrender.

**The fix:** a new rendering module (`phonics_ocr.shaping`) uses HarfBuzz
and FreeType — what a real typesetting system uses. It reads the OpenType
tables inside the font itself, so a harakah lands exactly where the type
designer put it. All eight faces now render correctly, including two the
old check had to reject.

A consequence: the **Naskh training run was stopped at epoch 1 of 20**,
because it would have trained on wrong data. It now deserves to be
repeated with the renderer fixed, and that is the first technical step
proposed below.

## 6. Verified operational facts

| Fact | Evidence |
|---|---|
| GPU inference works | Real execution on an RTX 3050 Ti, CUDA 11.8 |
| CPU fallback works | Verified; needed oneDNN disabled to work around a library bug |
| GPU is ~5x faster | 0.686 s against 3.449 s on the same image |
| Runs fully offline | Inference succeeded with the network blocked — zero connection attempts |
| Small enough for embedded hardware | 21 MB for all three models |
| Diacritics are in the model's charset | 747 characters, 38 of them Arabic marks — all eight classical harakat |
| Word-level boxes | **Not available from the detector** — it returns lines, not words |

That last row has a product consequence: highlighting one specific word on
the page requires a separate line-to-word segmentation step. That step is
written and tested, but it derives from aligning the model's own output
rather than from true word detection — a distinction that should not be
presented as real word detection.

## 7. What is not yet proven

- **The 95% target is not evaluated.** There is no real held-out test
  data. Every number in this report comes either from generated pages or
  from a *single* hand-labelled photograph.
- **There is no photographed test set.** 20–50 photographed and
  hand-transcribed pages are needed before any accuracy figure is
  statistically meaningful. This is the highest-value step in the project
  right now.
- **Arabic word accuracy is low** (43.6%) even on ideal input.
- **Diacritic recall swings** between 0.05 and 0.92 for reasons not yet
  understood.
- **A licensing matter.** The vowelled corpus used in training is GPL-2.0
  licensed and needs legal review before shipping inside a patented
  product.

## 8. Next steps, in priority order

1. **Photograph and transcribe 20–50 real pages.** Without them there is
   no accuracy figure to quote. Nothing in this report generalises to
   camera input.
2. **Regenerate the training data with the correct renderer** and repeat
   the Naskh training run that was stopped. This is the nearest expected
   improvement.
3. **Measure diacritic recall properly:** 50 vowelled pages, labelled both
   with and without tashkeel, so the gap is attributable to its cause.
4. **Find the resolution floor** at which a fatha stops being resolvable.
   This sets a hard requirement on the device's camera.
5. **Settle word localisation** (§6), since highlighting a word on the
   page depends on it.
6. **Review the corpus licence.**

---

## Reproducing the numbers in §3

```bat
python scripts\make_demo_page.py
python scripts\make_demo_video.py --device gpu
python -m pytest tests\ -q
```

The per-page figures come from `make_demo_video.py`'s own scoring, which
compares against the `expected_strings` in each page's `.json`. §4 comes
from `docs/finetuning.md`; §5 and §6 from `docs/baseline_findings.md`
§4b and §4.
