# Messages for collecting the data

Replace the `[...]` parts before sending. Give each person **one** form link and **one**
contributor code (P01, P02, ...), and keep a private list of who has which code.

## 1. WhatsApp: friends, classmates, colleagues (Egyptian Arabic)

> أهلاً [الاسم] 👋
> عندي طلب صغير لمشروع الكورس بتاعي في الـ NLP 🙏
> المشروع مساعد بيفهم رسايل بالمصري وبيعمل تذكير أو منبه، والمطلوب ناس تكتب رسايل **بطريقتها الطبيعية**، زي ما بتكتب لأي حد على الموبايل.
>
> الفورم فيها مواقف قصيرة (زي "عايز تفتكر تبعت الـ report للمدير")، وإنت تكتب الرسالة اللي كنت هتبعتها. حوالي **25 دقيقة**.
>
> 🔗 [لينك الفورم]
> 🔑 الكود بتاعك: **[P0X]** (اكتبه في خانة الكود في أول الفورم)
>
> ملحوظتين مهمين: متستخدمش ChatGPT أو أي ترجمة، كلامك إنت بس هو اللي مهم 😊 وأي أسماء تكتبها تكون من خيالك.
> لو ينفع تخلص لحد [اليوم] هتفرق جداً. شكراً!

## 2. LinkedIn or PM groups (project managers)

> **Egyptian project managers: 25 minutes to help a university NLP project?**
>
> I'm building ATHAR Chat, an assistant that understands work messages in Egyptian Arabic
> ("فكرني أبعت الـ status report بكرة الصبح") and sets one reminder or alarm. To test it fairly I
> need real messages written by Egyptian PMs, the way you actually type: standups, client
> follow-ups, Jira, UAT, go-lives.
>
> The form shows short situations and you write the message you'd send. No AI tools, just your
> own wording. Only a contributor code is stored, not your name.
>
> Comment or DM me and I'll send you a form link. Thank you! 🙏

## 3. Reminder, 3 days later (Egyptian Arabic)

> أهلاً [الاسم]، بفكرك بالفورم بتاع المشروع 🙏 لو لسه ما لحقتش، ممكن تعمل نصها بس وتبعت، أي رسايل بتفرق. الكود: [P0X]، اللينك: [لينك]. شكراً!

## 4. Email to the instructor (dataset design change)

> **Subject:** ATHAR Chat: proposed change to the dataset composition
>
> Dear [Dr. ...],
>
> My proposal says the 300 dataset items are written by native Egyptian Arabic speakers.
> Collecting 300 native requests in the time left is the main risk to the project, so I would
> like your approval for this design:
>
> - **Test (120 items) and, as far as possible, dev (40) are native-written**, by me and 5-8
>   contributors (mostly project managers), from scenario cards with no example wording. The
>   test split is chosen by the existing family-level splitter and stays locked until one final run.
> - **Training data may include the 300 assistant-drafted items** I prepared for the pipeline
>   dry runs. They are labelled `source: assistant_draft` and the split tool already refuses
>   them in test.
> - The report will state this openly, report results for train-on-drafts / test-on-native,
>   and keep the native out-of-distribution check (200 human-written Egyptian commands from a
>   public CC BY 4.0 dataset).
>
> The hypotheses, metrics and statistics stay as pre-registered. The change will be logged in
> the protocol's deviations log, dated before any native data is collected.
>
> Would this be acceptable? I'm happy to discuss.
>
> Best regards,
> [Name]
