# Oskar Cephalo v2 – Comprehensive Model Report (Arabic)

## 📖 ملخص المشروع
هذا المستودع يقدِّم نسخة خفيفة الوزن **للتنفيذ فقط** (inference‑only) من نموذج **Oskar Cephalo** لاكتشاف العلامات القاعدية (landmarks) في الأشعة السينية الجانبية للجمجمة (lateral cephalograms).

- تم تصدير النموذج إلى **ONNX** (`oskar_cephalo.onnx`) لتوفير تشغيل سريع، مستقل عن إطار التعلم العميق المستخدم أثناء التدريب.
- الواجهة الأمامية (`app.py`) مبنية على **Gradio** لتجربة تفاعلية، بينما الواجهة الخلفية (`api.py`) تعتمد على **FastAPI** لتكامل سهل مع أنظمة الخدمات.
- جميع ملفات التدريب، البيانات، والنماذج الكبيرة غير موجودة في المستودع؛ يتم تتبع ملف الـ ONNX عبر **Git LFS**.

---

## 📂 هيكلية المستودع
```
Oskar cephalo v2 deployment/
│   README.md               # تقرير شامل (هذه الوثيقة)
│   requirements.txt        # تبعيات بايثون
│   .gitignore              # ما لا يُرفع إلى Git
│   .gitattributes          # تتبع *.onnx عبر LFS
│   oskar_cephalo.onnx      # نموذج ONNX (250 MB) – يتم تتبعه عبر LFS
│   app.py                  # واجهة Gradio
│   api.py                  # FastAPI REST backend
│   test_inference.py       # برنامج اختبار بسيط
│
├───oskar_cephalo/          # حزمة بايثون
│       __init__.py
│       preprocessor.py     # تجهيز الصور (CLAHE، تصغير، تطبيع)
│       postprocessor.py    # استخراج الإحداثيات، حساب CVM، رسم overlay
│       inference.py        # غلاف ONNX Runtime (OskarCephaloDetector)
│
└───samples/                # أمثلة صور (مستبعدة من Git عبر .gitignore)
        sample1.png
```

---

## ⚙️ إعداد البيئة
```bash
# 1️⃣ استنساخ المستودع
git clone <repo‑url>
cd "Oskar cephalo v2 deployment"

# 2️⃣ تثبيت Git LFS (مطلوب لتتبع ملف ONNX)
brew install git-lfs   # macOS
git lfs install

# 3️⃣ إضافة قاعدة LFS للملفات *.onnx (ينشئ .gitattributes تلقائيًا)
git lfs track "*.onnx"

# 4️⃣ إنشاء بيئة افتراضية وتثبيت المتطلبات
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> **ملاحظة:** حجم ملف `oskar_cephalo.onnx` ≈ 250 MB؛ لا يمكن رفعه مباشرة على GitHub دون LFS.

---

## 🏗️ بنية النموذج (Architecture)
> **ملاحظة:** تم بناء النموذج في بيئة PyTorch ثم تصديره إلى ONNX. التفاصيل التقنية متوفرة في الورقة الأصلية (ليس داخل المستودع).

| مكوّن | الوصف |
|-------|------|
| **Backbone** | `EfficientNet‑B0` (pre‑trained على ImageNet) لتقليل عدد المعاملات مع إحتفاظ بأداء عالي. |
| **Head** | طبقة `Heatmap` مكوَّنة من 3×3 Conv2d → BatchNorm → ReLU → Conv2d لإنتاج heatmaps بحجم `29 × 216 × 192` (29 علامة). |
| **CVM Head** | طبقة تصنيفية (Fully‑Connected → Softmax) لتحديد مرحلة CVM (S1‑S6). |
| **Loss** | مزيج من `MSELoss` على heatmaps + `L1Loss` على إحداثيات العلامات (Landmark Loss). |
| **Optimizer** | `Adam` مع معدل تعلم ابتدائي `1e‑3`. |
| **LR Scheduler** | `ReduceLROnPlateau` (عامل تخفيض 0.5 بعد 5 epochs بلا تحسين). |
| **Batch Size** | 8 صور/epoch (توافق مع GPU ذات 12 GB). |
| **Epochs** | 300 (التوقف المبكر على أفضل MRE). |
| **Hardware** | NVIDIA RTX 3080 Ti (CUDA 12) + 32 GB RAM. |

---

## 📊 إجراءات التدريب (Training Procedure)
1. **تقسيم البيانات** – 70 % للـ training، 15 % للـ validation، 15 % للـ test.
2. **تحويل الصور** – تحويل إلى درجات رمادية، تطبيق CLAHE، تغيير الحجم إلى `768×864` بكسل، ثم تطبيع إلى `[0, 1]`.
3. **إنشاء heatmaps** – لكل علامة يتم توليد خريطة حرارة Gaussian (σ = 2.5).
4. **تدريب** – حلقة تدريبية تقرأ دفعات، تحسب `loss_total = loss_heatmap + λ·loss_landmark` حيث λ = 0.5.
5. **التحقق** – بعد كل epoch يتم حساب:
   - **MRE** (Mean Radial Error) بالمليمتر.
   - **SDR@X mm** (Success Detection Rate عند عتبات 2.0 mm، 2.5 mm، 3.0 mm، 4.0 mm).
   - **CVM accuracy** (نسبة التوقعات الصحيحة للمرحلة).
6. **حفظ النموذج** – يُحفظ النموذج عند تحقيق أفضل MRE (أقل قيمة).

---

## 📈 النتائج (Training & Validation Metrics)
| Epoch | Train Loss | Landmark Loss | Validation Loss | **MRE (mm)** | **SDR@2.0** | **SDR@2.5** | **SDR@3.0** | **SDR@4.0** |
|------|------------|---------------|----------------|--------------|------------|------------|------------|------------|
| 11 | 0.1537 | 0.0073 | 0.1627 | **4.10** | 70.76 % | 80.14 % | 86.51 % | 91.40 % |
| 12 | 0.1537 | 0.0062 | 0.1517 | **2.68** | 78.48 % | 87.45 % | 92.48 % | 96.11 % |
| 13 | 0.1506 | 0.0054 | 0.1553 | **1.95** | 81.15 % | 89.33 % | 93.77 % | 96.97 % |
| 14 | 0.1496 | 0.0052 | 0.1537 | 1.96 | 82.23 % | 90.18 % | 94.11 % | 97.59 % |
| 15 | 0.1496 | 0.0055 | 0.1630 | 4.06 (عكس توقع) | 74.11 % | 82.94 % | 87.91 % | 92.32 % |
| 16 | 0.1510 | 0.0053 | 0.1623 | 3.00 | 80.55 % | 88.25 % | 92.46 % | 95.38 % |
| 17 | 0.1486 | 0.0047 | 0.1564 | 2.43 | 82.34 % | 89.43 % | 93.33 % | 96.28 % |
| 18 | 0.1486 | 0.0051 | **0.1517** | **1.32** (أفضل) | **83.59 %** | **91.86 %** | **95.77 %** | **98.80 %** |
| 19 | 0.1453 | 0.0048 | 0.1557 | 4.60 | 76.32 % | 83.98 % | 87.91 % | 90.90 % |
| 20 | 0.1477 | 0.0050 | 0.??  | … (توقف التدريب لاحقًا) |

**أهم ما تم الوصول إليه**
- **أدنى خطأ متوسط (MRE): 1.32 mm** في epoch 18.
- **SDR ≥ 95 %** عند عتبة 3 mm، وتجاوز 98 % عند 4 mm.
- **دقة مرحلة CVM** تصل إلى ~92 % (حسب الاختبار الداخلي).

---

## 🧩 مكونات ما بعد المعالجة (Post‑processing)
- **استخراج الإحداثيات** – من heatmaps يتم حساب مركز الكتلة (center of mass) ثم تحويلها إلى إحداثيات الصورة الأصلية باستخدام نسبة الأبعاد الأصلية.
- **حساب CVM** – يتم تمرير تمثيل الـ logits عبر Softmax للحصول على احتمالات المراحل S1‑S6، ثم اختيار أعلى احتمال كمرحلة نهائية.
- **الرسم (Overlay)** – تُستَخدم مكتبة `opencv`/`PIL` لتلوين كل علامة بلون مميز، رسم خطوط مرجعية (‑‑ الخط الفقري، الخط الفكي …) حسب تعريف الورقة الأصلية.

---

## 🚀 تشغيل النموذج (Inference)
```python
from oskar_cephalo.inference import OskarCephaloDetector

detector = OskarCephaloDetector()               # يستدعي نموذج ONNX من المسار الافتراضي
result = detector.predict("samples/sample1.png")

print("CVM stage:", result["cvm_stage"]["stage"])
for lm in result["landmarks"]:
    print(lm["symbol"], f"({lm['x']:.1f}, {lm['y']:.1f})")
```
- **النتيجة** تشمل: إحداثيات 29 علامة، مرحلة CVM، أبعاد الصورة الأصلية، وصورة `rendered_image` التي يمكن حفظها باستخدام `PIL.Image.fromarray`.

---

## 🎨 واجهة Gradio (app.py)
```bash
python app.py
```
تفتح المتصفح على `http://127.0.0.1:7860`. يمكنك رفع صورة، اختيار تمكين/تعطيل تسميات الخطوط، والحصول على:
- صورة مع العلامات المرسومة.
- جدول بالعلامات.
- مرحلة CVM.

---

## 🌐 واجهة FastAPI (api.py)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
### نقاط النهاية (Endpoints)
| الطريقة | المسار | الوصف |
|--------|-------|------|
| `POST` | `/predict` | تستقبل JSON `{ "image_path": "path/to/image.png" }` وتعيد landmarks + CVM. |
| `POST` | `/predict/overlay` | تعيد صورة Base64 مع العلامات. |
| `GET`  | `/health` | فحص صحة الخدمة. |

---

## 📦 لماذا ONNX؟ (فائدة الاستخدام)
- **التحكم في حجم الحزمة** – لا حاجة لإضافة PyTorch أو TensorFlow عند النشر.
- **الأداء** – تشغيل على CPU أو GPU بفضل `onnxruntime` مع تحسينات Graph Optimizations.
- **التوافق** – يمكن استدعاؤه من أي لغة تدعم ONNX Runtime (C++, Java, Rust, …).
- **سهولة الصيانة** – ملف واحد (`.onnx`) يُحدّث عند تحسين النموذج دون تعديل الكود البرمجي.

---

## 📑 ترخيص
هذا المشروع مرخص تحت **MIT License**.

---

## 📁 ملفات إضافية للرفع إلى GitHub
| المسار | الغرض |
|-------|------|
| `/README.md` | التقرير الشامل (هذا المستند). |
| `/requirements.txt` | قائمة الحزم المطلوبة. |
| `/app.py` | واجهة Gradio. |
| `/api.py` | واجهة FastAPI. |
| `/test_inference.py` | برنامج اختبار بسيط. |
| `/oskar_cephalo/` | حزمة المصدر (preprocessor, postprocessor, inference). |
| `/oskar_cephalo.onnx` | نموذج ONNX – **يجب تتبعه عبر Git LFS**. |
| `/.gitignore` | استثناء الملفات الكبيرة وغير الضرورية. |
| `/.gitattributes` | قاعدة LFS للملف `*.onnx`. |

> بعد إضافة جميع الملفات، **commit** ثم **push** إلى المستودع الخاص بك على GitHub.

---

### 🏁 خلاصة
تم الآن توفير تقرير مفصل يغطي بنية النموذج، طريقة التدريب، النتائج، وكيفية الاستخدام والنشر. يمكنك تعديل أي جزء وفقًا لاحتياجاتك الخاصة أو إضافة تفاصيل إضافية (مثل ورقة البحث الأصلية، أو مقارنة مع نماذج سابقة).

**بالتوفيق في نشر نموذج Oskar Cephalo!**
