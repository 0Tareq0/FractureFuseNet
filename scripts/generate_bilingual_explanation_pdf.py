import os
import subprocess
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["scripts", "phase 1"] else SCRIPT_DIR
if (ROOT_DIR / "phase 1").exists():
    PHASE1_DIR = ROOT_DIR / "phase 1"
else:
    PHASE1_DIR = ROOT_DIR

html_content = """<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="utf-8">
<title>FractureFuseNet: Phase 1 Bilingual Technical Guide (সহজ বাংলা ও ইংরেজি ব্যাখ্যা)</title>
<style>
  @page {
    size: A4;
    margin: 12mm 12mm 12mm 12mm;
  }

  body {
    font-family: 'Segoe UI', 'Vrinda', 'Kalpurush', 'Noto Sans Bengali', Arial, sans-serif;
    color: #1e293b;
    line-height: 1.52;
    font-size: 8.8pt;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }

  .header-card {
    border-bottom: 2.5px solid #1d4ed8;
    padding-bottom: 8px;
    margin-bottom: 11px;
  }

  .header-badge {
    background: #dbeafe;
    color: #1e40af;
    padding: 2.5px 9px;
    border-radius: 9999px;
    font-size: 7.8pt;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  h1 {
    color: #0f172a;
    font-size: 15pt;
    margin: 2px 0 5px 0;
    font-weight: 700;
  }

  .meta-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 8.2pt;
    color: #475569;
    margin-top: 5px;
  }

  .meta-item {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    padding: 2.5px 8px;
    border-radius: 4px;
  }

  .meta-item strong {
    color: #0f172a;
  }

  h2 {
    color: #1e3a8a;
    font-size: 11pt;
    margin-top: 13px;
    margin-bottom: 5px;
    border-left: 4px solid #2563eb;
    padding-left: 7px;
    page-break-after: avoid;
  }

  h3 {
    color: #1e293b;
    font-size: 9.6pt;
    margin-top: 9px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }

  p {
    margin: 4px 0;
    text-align: justify;
  }

  ul, ol {
    margin: 4px 0;
    padding-left: 17px;
  }

  li {
    margin-bottom: 3px;
  }

  .alert {
    border-radius: 5px;
    padding: 8px 11px;
    margin: 8px 0;
    font-size: 8.6pt;
    page-break-inside: avoid;
  }

  .alert-warning {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    color: #92400e;
  }

  .alert-info {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    color: #1e3a8a;
  }

  .alert-bangla {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    color: #14532d;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 7px 0;
    font-size: 7.9pt;
    page-break-inside: avoid;
  }

  th {
    background-color: #f1f5f9;
    color: #1e293b;
    font-weight: 700;
    text-align: left;
    padding: 4.5px 6px;
    border: 1px solid #cbd5e1;
  }

  td {
    padding: 4px 6px;
    border: 1px solid #e2e8f0;
    vertical-align: middle;
  }

  tr:nth-child(even) {
    background-color: #f8fafc;
  }

  .badge-passed {
    background-color: #dcfce7;
    color: #166534;
    padding: 1.5px 5.5px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 7.2pt;
  }

  .badge-warn {
    background-color: #fef3c7;
    color: #92400e;
    padding: 1.5px 5.5px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 7.2pt;
  }

  pre, code {
    font-family: 'Consolas', 'Courier New', monospace;
  }

  pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 7px 9px;
    border-radius: 4px;
    font-size: 7.5pt;
    margin: 5px 0;
    line-height: 1.34;
    page-break-inside: avoid;
  }

  .code-inline {
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 1px 3.5px;
    border-radius: 3px;
    font-size: 7.8pt;
    border: 1px solid #e2e8f0;
  }

  .footer-note {
    margin-top: 14px;
    border-top: 1px solid #cbd5e1;
    padding-top: 5px;
    font-size: 7.5pt;
    color: #64748b;
    display: flex;
    justify-content: space-between;
    page-break-inside: avoid;
  }

  .page-break {
    page-break-after: always;
  }

  .q-card {
    background: #f8fafc;
    border: 1.5px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 8px 0;
    page-break-inside: avoid;
  }

  .q-title {
    color: #1e3a8a;
    font-weight: 700;
    font-size: 9.8pt;
    margin-bottom: 4px;
  }
</style>
</head>
<body>

<!-- PAGE 1 -->
<div class="header-card">
  <span class="header-badge">CSE438 • Project 12 • Bilingual Explanatory Guide</span>
  <h1>FractureFuseNet: Phase 1 Summary & Architectural Changes Explanation</h1>
  <div class="meta-grid">
    <div class="meta-item"><strong>বিষয়:</strong> Phase 1 এর কাজ ও আর্কিটেকচারাল পরিবর্তনের সহজ বাংলা ও ইংরেজি ব্যাখ্যা</div>
    <div class="meta-item"><strong>ডেটাসেট:</strong> HBFMID (1,539 Images / 1,429 Clean Single-Label Included)</div>
    <div class="meta-item"><strong>উদ্দেশ্য:</strong> শিক্ষক ও ভাইভা প্রেজেন্টেশনে সহজে আত্মবিশ্বাসের সাথে উপস্থাপন করার জন্য</div>
  </div>
</div>

<h2>পর্ব ১: Phase 1-এ আমরা কী কী কাজ করেছি? (Very Brief Summary)</h2>
<p>
সংক্ষেপে Phase 1-এ আমরা কাঁচা ডেটাসেটের ভুলত্রুটি ও মারাত্মক ডেটা লিকেজ দূর করে ডিপ লার্নিংয়ের জন্য একটি বৈজ্ঞানিকভাবে ১০০% নিখুঁত ও নির্ভরযোগ্য প্রি-প্রসেসিং পাইপলাইন তৈরি করেছি:
</p>
<ol>
  <li><strong>ডেটাসেট অডিট (Dataset Integrity Audit):</strong> মোট <strong>১,৫৩৯টি ছবি</strong> (৬৪০×৬৪০ RGB) এবং <strong>১,৫৩৯টি YOLO লেবেল</strong> (১,৭৭১টি বাউন্ডিং বক্স) সম্পূর্ণ পরীক্ষা করা হয়েছে। কোনো ফাইল নষ্ট বা করাপ্ট ছিল না এবং কোঅর্ডিনেটগুলো নিখুঁত ছিল।</li>
  <li><strong>লেবেল ক্লিনিং ও ফিল্টারিং (Annotation Disambiguation):</strong> যে ছবিগুলোতে একাধিক বাউন্ডিং বক্স ভিন্ন ভিন্ন ক্লাসের ছিল, সেগুলোকে ফিল্টার করা হয়েছে। <strong>১,৪২৯টি পরিষ্কার ছবি</strong> (1,423 unanimous + 6 strict majority) রাখা হয়েছে এবং ১১০টি বিতর্কিত রেকর্ড (107 multi-class ties + 3 খালি লেবেল ফাইল) স্প্লিট থেকে বাদ দেওয়া হয়েছে।</li>
  <li><strong>রোবোফ্লোর ৪০.৮% ডেটা লিকেজ সমাধান (Duplicate Leakage Elimination):</strong> রোবোফ্লোর মূল স্প্লিটে একই হাড়ের ফ্লিপ/রোটেট করা ছবি ট্রেন ও টেস্টে ছড়িয়ে ছিল। আমরা <strong>Dihedral Perceptual Hashing (<span class="code-inline">pHash</span> + <span class="code-inline">dHash</span> ≤ 10)</strong> ব্যবহার করে <strong>৬৫৭টি মূল সোর্স ক্লাস্টার</strong> শনাক্ত করেছি (যা মেন্ডেলি পেপারের মূল ৬৪১টি এক্স-রের সাথে মিলে যায়)।</li>
  <li><strong>গ্রুপ-অ্যাওয়ার ৬০:২০:২০ স্প্লিট (Zero-Leakage Splitting):</strong> ২৩৫টি কম্পোজিট গ্রুপের উপর ৫-ফোল্ড <span class="code-inline">StratifiedGroupKFold</span> করে ট্রেন (৮৫৭), ভ্যালিডেশন (২৮৫) এবং টেস্ট (২৮৭) ভাগ করা হয়েছে। ট্রেন ও টেস্টের মধ্যে <strong>ডেটা লিকেজ শূন্য (0.00%)</strong> প্রমাণিত।</li>
  <li><strong>সেগমেন্টাল ক্লাসের গাণিতিক সীমাবদ্ধতা প্রমাণ (Pigeonhole Principle):</strong> <span class="code-inline">Segmental</span> ফ্র্যাকচারের মাত্র ৮টি ছবি ছিল যা মাত্র ২টি সোর্স গ্রুপের অন্তর্ভুক্ত। ২টি গ্রুপকে কোনোভাবেই লিকেজ ছাড়া ৩টি সেটে ভাগ করা সম্ভব নয়। তাই টেস্টে ০ রেখে ৫-ফোল্ড ক্রস ভ্যালিডেশনের মাধ্যমে এর ১০০% ছবি মূল্যায়নের ব্যবস্থা করা হয়েছে।</li>
  <li><strong>মোডালিটি অডিট (X-ray vs MRI Confounding):</strong> ডেটাসেটের ৬৪.৬% Comminuted ফ্র্যাকচার ছিল MRI, যেখানে ৫টি ক্লাসে কোনো MRI ছিল না। মডেল যেন MRI-এর কালো টেক্সচার দেখে ফ্র্যাকচার অনুমান না করে, সেজন্য <strong>এক্স-রে (১,২২৭টি ছবি) মূল বেঞ্চমার্ক</strong> নির্ধারণ করা হয়েছে।</li>
  <li><strong>৬৪-মাত্রার ক্ল্যাসিক্যাল ফিচার নিষ্কাশন (64-dim Feature Vector):</strong> পূর্ণ রেজোলিউশন (৬৪০×৬৪০) ছবি থেকে CLAHE, মাল্টি-থ্রেশহোল্ড Canny edge, Hough transform lines, সার্কুলার অ্যাঙ্গেল ডাবলিং ($2\theta$), এবং Sobel গ্রেডিয়েন্ট ফিচার বের করা হয়েছে।</li>
  <li><strong>পরিসংখ্যানগত সিউডোরিপ্লিকেশন সমাধান (Cluster-Level Permutation Test):</strong> ডুপ্লিকেট অগমেন্টেড ছবিতে টেস্ট করার কারণে p-value ১.৫ কোটি গুণ অতিরঞ্জিত হচ্ছিল। আমরা ৪৯৬টি ক্লাস্টারের মিডিয়ান নিয়ে ১০,০০০ পারমিউটেশন ও Holm/BH কারেকশন করেছি।</li>
  <li><strong>পাইটর্চ মাল্টি-মোডাল পাইপলাইন ও অটোমেশন:</strong> শেয়ার্ড জিওমেট্রিক অগমেন্টেশন, শুধুমাত্র ট্রেনের উপর স্কেলার ফিটিং (SHA-256 প্রোভেন্যান্স), ফোকাল লস (Focal Loss), ১৬/১৬ ইউনিট টেস্ট এবং ১২/১২ অটোমেটেড পাইপলাইন রানার তৈরি করা হয়েছে।</li>
</ol>

<hr style="border: 0; border-top: 1px solid #cbd5e1; margin: 10px 0;">

<h2>পর্ব ২: ছবিকে কেন 3 × 224 × 224 টেনসরে নেওয়া হলো? এবং এখানে "3" মানে কী?</h2>

<div class="q-card">
  <div class="q-title">১. এখানে "3" (Three) মানে কী? এক্স-রে তো সাদা-কালো (Grayscale), তাহলে ৩ কেন?</div>
  <p>
    পাইটর্চে (PyTorch) যে কোনো ২ডি কনভোলিউশনাল লেয়ার ছবিকে ৩টি ডাইমেনশনের টেনসর আকারে গ্রহণ করে:
    <strong>Tensor Shape = (Channels, Height, Width) = (C, H, W)</strong>। এখানে <strong>3 মানে ৩টি কালার চ্যানেল: Red, Green, এবং Blue (RGB)</strong>।
  </p>
  <p>
    <strong>খুব স্বাভাবিক প্রশ্ন:</strong> মেডিকেল এক্স-রে বা এমআরআই তো সাদা-কালো (Grayscale)! তাহলে ১ চ্যানেল না দিয়ে আমরা ৩ চ্যানেল কেন দিলাম?
  </p>
  <p>
    <strong>বৈজ্ঞানিক ও টেকনিক্যাল কারণ (Pretrained Transfer Learning):</strong><br>
    আমরা ডিপ লার্নিংয়ের যেসব শক্তিশালী মডেল ব্যবহার করব (যেমন <strong>ResNet-50, EfficientNet-B0, DenseNet-121, ConvNeXt</strong>), সেগুলো ImageNet ডেটাসেটের ১২ লাখ রঙিন ছবির উপর প্রি-ট্রেইন করা। এই মডেলগুলোর প্রথম কনভোলিউশনাল লেয়ার (<span class="code-inline">conv1</span>) ইনপুট হিসেবে বাধ্যতামূলকভাবে <strong><span class="code-inline">in_channels = 3</span></strong> প্রত্যাশা করে:
  </p>
  <pre>nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)</pre>
  <ul>
    <li>আমরা যদি ১-চ্যানেলের সাদা-কালো ছবি <span class="code-inline">(1, 224, 224)</span> দিই, তাহলে পাইটর্চ এরর দেবে: <em>"Expected 3 channels, got 1"</em>।</li>
    <li>আর যদি প্রথম লেয়ারটি জোর করে বদলে ১-চ্যানেল করে দিই, তাহলে ImageNet-এর শেখা <strong>সব ফিল্টার ও ওজন (Weights) মুছে যাবে এবং নতুন করে র‍্যান্ডম ইনিশিয়ালাইজেশন হবে</strong> (ফলে ট্রান্সফার লার্নিংয়ের বিশাল সুবিধা আমরা হারাবো)।</li>
    <li><strong>আমাদের সমাধান:</strong> আমরা একই সাদা-কালো এক্স-রে ছবিকে ৩টি চ্যানেলেই কপি ($R = G = B$) করে দিয়েছি। এর ফলে মডেলের কোনো লেয়ার নষ্ট না করেই ImageNet-এর শক্তিশালী প্রি-ট্রেইন্ড ফিচারগুলো হাড়ের ছবিতে সরাসরি ব্যবহার করা যাচ্ছে!</li>
  </ul>
</div>

<div class="q-card">
  <div class="q-title">২. ছবির সাইজ ৬৪০×৬৪০ থেকে কেন 224 × 224 করা হলো?</div>
  <p>
    আমাদের ডেটাসেটের মূল ছবিগুলো ছিল ৬৪০×৬৪০ পিক্সেলের। সেগুলোকে ডিপ সিএনএন ব্রাঞ্চের জন্য ২২৪×২২৪ করার তিনটি গুরুত্বপূর্ণ কারণ রয়েছে:
  </p>
  <ol>
    <li><strong>প্রি-ট্রেইন্ড মডেলের স্ট্যান্ডার্ড ইনপুট:</strong> ResNet, DenseNet এবং EfficientNet মূলত ২২৪×২২৪ সাইজের ছবির জন্যই অপ্টিমাইজড এবং টিউন করা।</li>
    <li><strong>মেমোরি ও স্পিড (Computational Efficiency):</strong>
      <ul>
        <li>একটি ৬৪০×৬৪০ ছবিতে থাকে <strong>৪০৯,৬০০ পিক্সেল</strong>।</li>
        <li>একটি ২২৪×২২৪ ছবিতে থাকে <strong>৫০,১৭৬ পিক্সেল</strong>। অর্থাৎ ছবির মেমোরি ও গণনার পরিমাণ <strong>৮ গুণেরও বেশি (8.16×) কমে যায়</strong>!</li>
        <li>৬৪০×৬৪০ ছবি ডিপ মডেলে দিলে GPU VRAM ক্র্যাশ করত এবং ব্যাচ সাইজ ২ বা ৪ রাখতে হতো (যা ব্যাচ নরমালাইজেশনকে ধ্বংস করে)। ২২৪×২২৪ সাইজে আমরা স্বাচ্ছন্দ্যে ১৬ বা ৩২ ব্যাচ সাইজে ট্রেইনিং চালাতে পারব।</li>
      </ul>
    </li>
    <li><strong>সবচেয়ে গুরুত্বপূর্ণ প্রশ্ন: ২২৪×২২৪ করলে কি সূক্ষ্ম ফাটল (Hairline Fracture) হারিয়ে যাবে না?</strong>
      <ul>
        <li><strong>উত্তর:</strong> হ্যাঁ, সাধারণ সিএনএন মডেলে ছবি ছোট করলে সূক্ষ্ম ফাটল ব্লার হয়ে যায়। <strong>আর ঠিক এই সমস্যা সমাধানের জন্যই আমাদের FractureFuseNet-এর সৃষ্টি!</strong></li>
        <li>আমাদের মডেলে রয়েছে <strong>দুটি সমান্তরাল ব্রাঞ্চ (Dual-Branch Late Fusion)</strong>:
          <ul>
            <li><strong>Branch A (Deep CNN):</strong> ছবিটি ছোট করে <strong>২২৪×২২৪</strong> আকারে দেখে, যাতে সে হাড়ের সামগ্রিক গঠন (এটা কি হাতের হাড় নাকি পায়ের হাড়) বুঝতে পারে।</li>
            <li><strong>Branch B (Classical Edge/Hough):</strong> ছবিটি ছোট করে না! সে সরাসরি মূল সাইজের <strong>৬৪০×৬৪০ হাই-রেজোলিউশন ছবিতে</strong> Canny Edge ও Hough Line চালিয়ে চুল পরিমাণ সূক্ষ্ম ফাটলও পিক্সেল লেভেলে নিখুঁতভাবে শনাক্ত করে ৬৪টি ফিচারে রূপান্তর করে।</li>
          </ul>
        </li>
        <li>মডেলের শেষে এই দুটো ব্রাঞ্চকে যুক্ত (Late Fusion) করা হয়। ফলে আমরা সামগ্রিক অ্যানাটমি এবং সূক্ষ্ম ফাটলের নিখুঁত তথ্য — দুটোই একই সাথে পাই!</li>
      </ul>
    </li>
  </ol>
</div>

<div class="page-break"></div>

<!-- PAGE 2 -->
<h2>পর্ব ৩: অন্য কী কী গুরুত্বপূর্ণ পরিবর্তন করা হয়েছে এবং কেন? (Other Changes & The Scientific "Why")</h2>

<div class="q-card">
  <div class="q-title">১. শেয়ার্ড জিওমেট্রিক অগমেন্টেশন (Shared Geometric Transform) কেন আনা হলো?</div>
  <p>
    <strong>সমস্যা:</strong> আগে পাইটর্চের ডেটালোডারে যখন ছবি ফ্লিপ (অনুভূমিক উল্টানো) বা রোটেট হতো, তখন ডিপ লার্নিং ব্রাঞ্চ উল্টানো ছবি পাচ্ছিল, কিন্তু ক্ল্যাসিক্যাল এজ ব্রাঞ্চ পাচ্ছিল মূল সোজা ছবি! ফলে ফিউশনের সময় দুটি ব্রাঞ্চ হাড়ের দুটি বিপরীত দিক দেখে বিভ্রান্ত হচ্ছিল।
  </p>
  <p>
    <strong>সমাধান:</strong> আমরা <span class="code-inline">apply_shared_geometric_transform()</span> ফাংশন তৈরি করেছি। ট্রেইনিংয়ের সময় একটি ছবি ফ্লিপ বা রোটেট হলে, ডিপ ব্রাঞ্চ এবং এজ ব্রাঞ্চ—উভয়েই ঠিক <strong>একই সাথে এবং একই কোণে</strong> পরিবর্তিত ছবি পায়।
  </p>
</div>

<div class="q-card">
  <div class="q-title">২. ৬৪-ফিচার ভেক্টর থেকে উজ্জ্বলতা (Brightness) ও কনট্রাস্ট কেন বাদ দেওয়া হলো?</div>
  <p>
    <strong>সমস্যা:</strong> আগের স্ক্রিপ্টে ছবির গড় উজ্জ্বলতা (Mean Brightness) ও ভ্যারিয়েন্স ছিল। এক্সপ্লোরেটরি অ্যানালাইসিসে আমরা দেখলাম যে MRI স্ক্যানগুলো X-ray এর চেয়ে প্রাকৃতিকভাবেই অনেক বেশি কালো (Dark) হয়। মডেল তখন হাড়ের ফাটল না দেখে ছবি কতটা কালো বা সাদা তা দেখে সিদ্ধান্ত নেওয়া শুরু করেছিল (যাকে এআই-তে <em>Shortcut Learning</em> বা চিটিং বলা হয়)!
  </p>
  <p>
    <strong>সমাধান:</strong> আমরা ব্রাইটনেস পুরোপুরি বাদ দিয়ে CLAHE কন্ট্রাস্ট নরমালাইজেশন এনেছি এবং ফাটলের কৌণিক বিচ্ছুরণ ($2\theta$ সার্কুলার স্ট্যাটিস্টিকস) ও হেটারোজিনিটি যুক্ত করেছি। এখন মডেল কোনো শর্টকাট নিতে পারে না।
  </p>
</div>

<div class="q-card">
  <div class="q-title">৩. ডাইহেড্রাল পারসেপচুয়াল হ্যাশিং (Dihedral pHash/dHash) দিয়ে স্প্লিট কেন করা হলো?</div>
  <p>
    <strong>সমস্যা:</strong> রোবোফ্লোর মূল স্প্লিটে মারাত্মক ডেটা লিকেজ ছিল (৪০.৮২%)। একই হাড়ের ফ্লিপ বা রোটেট করা ৩টি ছবি একই সাথে ট্রেন ও টেস্ট সেটে রাখা হয়েছিল। পরীক্ষার প্রশ্ন ফাঁসের মতো মডেল আগেই টেস্টের ছবি দেখে ফেলায় টেস্ট অ্যাকুরেসি কৃত্রিমভাবে অনেক বেশি আসত! এছাড়া ফাইলের নাম <span class="code-inline">225_jpg</span> একই হওয়া সত্ত্বেও তার ভেতরে ভিন্ন ভিন্ন রোগীর হাড়ের ছবি মিশে ছিল।
  </p>
  <p>
    <strong>সমাধান:</strong> আমরা ছবির পিক্সেল প্যাটার্ন দেখে ৪টি কোণে (০°, H-Flip, V-Flip, ১৮০°) ডাইহেড্রাল হ্যাশিং করে ৬৫৭টি মূল ক্লাস্টার বের করেছি এবং গ্রুপ-অ্যাওয়ার স্প্লিট করেছি। এখন <strong>ট্রেন এবং টেস্টের মধ্যে কোনো মিল বা লিকেজ নেই (0.00% Leakage)</strong>।
  </p>
</div>

<div class="q-card">
  <div class="q-title">৪. স্কেলার ও নরমালাইজেশন কেন শুধুমাত্র ট্রেনিং সেটের (N=857) উপর ফিট করা হলো?</div>
  <p>
    <strong>সমস্যা:</strong> পুরো ডেটাসেট (Train + Valid + Test) মিলিয়ে Mean ও Std হিসাব করলে ভ্যালিডেশন এবং টেস্ট সেটের তথ্য ট্রেইনিংয়ে আগেই ফাঁস হয়ে যায়। একে মেশিন লার্নিংয়ের ভাষায় বলা হয় <strong>Data Snooping বা Lookahead Bias</strong>।
  </p>
  <p>
    <strong>সমাধান:</strong> আমরা শুধুমাত্র ৮৫৭টি ট্রেনিং ছবির উপর <span class="code-inline">StandardScaler</span> ফিট করেছি এবং টেস্ট সেটকে সম্পূর্ণ অদেখা রেখেছি। এছাড়াও SHA-256 ক্রিপ্টোগ্রাফিক হ্যাশ দিয়ে এর সততা নিশ্চিত করেছি।
  </p>
</div>

<div class="q-card">
  <div class="q-title">৫. সিউডোরিপ্লিকেশন (Pseudoreplication) সমাধান ও সুস্থ হাড়ের এজ বেশি পাওয়ার বৈজ্ঞানিক ব্যাখ্যা</div>
  <p>
    <strong>সমস্যা:</strong> একই হাড়ের অগমেন্টেড কপিগুলোর উপর আলাদা আলাদা স্ট্যাটিস্টিক্যাল টেস্ট চালালে p-value প্রায় দেড় কোটি গুণ ($1.5 \times 10^7\times$) অতিরঞ্জিত হয় (একে স্ট্যাটিস্টিক্সে <em>Pseudoreplication</em> বলে)।
  </p>
  <p>
    <strong>সমাধান ও বৈজ্ঞানিক পর্যবেক্ষণ:</strong> আমরা ৪৯৬টি মূল ক্লাস্টারে টেস্ট চালিয়েছি। একটি মজার বিষয় পাওয়া গেছে: <strong>সুস্থ হাড়ের ছবিতে এজ ও লাইনের সংখ্যা ফাটলযুক্ত ছবির চেয়ে বেশি!</strong> কেন? কারণ সুস্থ হাড়ের ছবিগুলো ছিল হাত বা কব্জির অনেকগুলো ছোট হাড়ের জোড়ের ছবি (Wrist/Carpal bones), যেখানে প্রাকৃতিকভাবেই অনেক লাইন থাকে। তবে ফাটলযুক্ত ছবিতে সনাক্ত হওয়া লাইনগুলোর দৈর্ঘ্য সুস্থ হাড়ের চেয়ে উল্লেখযোগ্যভাবে বড় (৫৭.৩৮ পিক্সেল বনাম ৪৮.৬০ পিক্সেল, $p = 3.65 \times 10^{-4}$)। এর দ্বারা প্রমাণিত হয় ক্ল্যাসিক্যাল ফিচারগুলো হাড়ের গভীর কাঠামোগত তথ্য সফলভাবে ধারণ করতে পারছে।
  </p>
</div>

<div class="q-card">
  <div class="q-title">৬. ফোকাল লস (Focal Loss) কেন ব্যবহার করা হলো?</div>
  <p>
    <strong>কারণ:</strong> ডেটাসেটে ক্লাস ইমব্যালেন্স অনেক বেশি। <span class="code-inline">Transverse Displaced</span> ফ্র্যাকচারের ছবি আছে ৫৬৮টি, অথচ <span class="code-inline">Linear</span> ফ্র্যাকচারের ছবি মাত্র ২২টি এবং <span class="code-inline">Segmental</span> মাত্র ৮টি! সাধারণ Cross-Entropy ব্যবহার করলে মডেল শুধু বড় ক্লাসের ছবি চিনত, ছোট ক্লাসগুলো ভুল করত। ফোকাল লস সহজ ছবির লস কমিয়ে দেয় এবং কঠিন ও ছোট ক্লাসের ফাটলগুলোর উপর মডেলকে বেশি মনোযোগ দিতে বাধ্য করে।
  </p>
</div>

<h2>পর্ব ৪: দ্রুত রিভিশন দেওয়ার জন্য তুলনা টেবিল (Quick Comparison Table)</h2>
<table>
  <thead>
    <tr>
      <th>বিষয় (Component)</th>
      <th>আগের কোড (Previous Naive)</th>
      <th>আমাদের সংশোধিত পাইপলাইন (Corrected Implementation)</th>
      <th>কেন এই পরিবর্তন করা হলো? (Scientific Reason)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ডিপ ইনপুট শেপ</strong></td>
      <td>অনির্দিষ্ট বা অমিল</td>
      <td><strong><span class="code-inline">torch.float32 [Batch, 3, 224, 224]</span></strong></td>
      <td>ImageNet প্রি-ট্রেইন্ড ব্যাকবোনের সাথে মিল রেখে ৩-চ্যানেল করা হয়েছে যাতে লেয়ার রি-ইনিশিয়ালাইজ না লাগে এবং মেমোরি বাঁচে।</td>
    </tr>
    <tr>
      <td><strong>ক্ল্যাসিক্যাল ইনপুট শেপ</strong></td>
      <td>রিসাইজড / ব্লার</td>
      <td><strong>Native (640 × 640) Grayscale + CLAHE</strong></td>
      <td>সূক্ষ্ম ফাটল যেন নষ্ট না হয়, সেজন্য ফুল রেজোলিউশনে Canny ও Hough চালানো হয়েছে।</td>
    </tr>
    <tr>
      <td><strong>অগমেন্টেশন</strong></td>
      <td>আলাদা র‍্যান্ডম ট্রান্সফর্ম</td>
      <td><strong><span class="code-inline">apply_shared_geometric_transform</span></strong></td>
      <td>ডিপ ব্রাঞ্চ ও ক্ল্যাসিক্যাল ব্রাঞ্চের মধ্যে একই অ্যাঙ্গেল ও ফ্লিপ বজায় রাখার জন্য।</td>
    </tr>
    <tr>
      <td><strong>ডুপ্লিকেট গ্রুপিং</strong></td>
      <td>ফাইলের নাম <span class="code-inline">base_id</span></td>
      <td><strong>Dihedral <span class="code-inline">pHash/dHash</span> ≤ 10 + Union-Find</strong></td>
      <td>৪০.৮২% লিকেজ দূর করে ৬৫৭টি মূল সোর্স ইমেজ ক্লাস্টার করার জন্য।</td>
    </tr>
    <tr>
      <td><strong>স্কেলার ফিটিং</strong></td>
      <td>পুরো ডেটাসেট দিয়ে ফিট</td>
      <td><strong>শুধুমাত্র ট্রেনিং সেট (N=857) + SHA-256 হ্যাশ</strong></td>
      <td>টেস্ট ডেটা ফাঁস (Lookahead Bias / Data Snooping) রোধ করার জন্য।</td>
    </tr>
    <tr>
      <td><strong>স্ট্যাটিস্টিক্স টেস্ট</strong></td>
      <td>অগমেন্টেড ইমেজ লেভেলে</td>
      <td><strong>৪৯৬টি ক্লাস্টার মিডিয়ান + ১০,০০০ পারমিউটেশন</strong></td>
      <td>সিউডোরিপ্লিকেশন দূর করে সঠিক ও বাস্তবসম্মত p-value পাওয়ার জন্য।</td>
    </tr>
    <tr>
      <td><strong>লস ফাংশন</strong></td>
      <td>সাধারণ Cross-Entropy</td>
      <td><strong>ক্লাস-ওয়েটেড ফোকাল লস (Focal Loss, γ=2.0)</strong></td>
      <td>বড় ক্লাসের আধিপত্য কমিয়ে দুর্লভ ফ্র্যাকচারগুলো সঠিকভাবে চেনার জন্য।</td>
    </tr>
  </tbody>
</table>

<div class="footer-note">
  <span>Course CSE438 • Project 12: FractureFuseNet • Phase 1 Bilingual Technical Guide</span>
  <span>Human Bone Fractures Multi-modal Image Dataset (HBFMID) • Published November 2026</span>
</div>

</body>
</html>
"""

html_path = str(PHASE1_DIR / "PHASE_1_BILINGUAL_EXPLANATION.html")
pdf_path = str(PHASE1_DIR / "PHASE_1_BILINGUAL_EXPLANATION.pdf")
root_html_path = str(ROOT_DIR / "PHASE_1_BILINGUAL_EXPLANATION.html")
root_pdf_path = str(ROOT_DIR / "PHASE_1_BILINGUAL_EXPLANATION.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

with open(root_html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

edge_candidates = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
edge_path = next((p for p in edge_candidates if os.path.exists(p)), edge_candidates[0])

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    html_path,
]

print("Compiling publication-quality Bilingual (English+Bengali) Explanation PDF via Microsoft Edge...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Edge Return code:", res.returncode)

if os.path.exists(pdf_path):
    file_size = os.path.getsize(pdf_path)
    print(f"SUCCESS: Clean Bilingual PDF generated at: {pdf_path}")
    print(f"File size: {file_size:,} bytes")
    shutil.copyfile(pdf_path, root_pdf_path)
    print(f"Copied to root directory: {root_pdf_path}")
else:
    raise RuntimeError(f"Failed to generate PDF: {res.stderr}")
