import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.parent.name == "phase 1":
    PHASE1_DIR = SCRIPT_DIR.parent
else:
    PHASE1_DIR = SCRIPT_DIR.parent / "phase 1"

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase 1 Technical Summary Report - FractureFuseNet</title>
<style>
  @page {
    size: A4;
    margin: 14mm 14mm 14mm 14mm;
  }

  body {
    font-family: 'Segoe UI', 'Vrinda', 'Kalpurush', 'Noto Sans Bengali', Arial, sans-serif;
    color: #1e293b;
    line-height: 1.48;
    font-size: 9.2pt;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }

  .header-card {
    border-bottom: 2.5px solid #1d4ed8;
    padding-bottom: 10px;
    margin-bottom: 14px;
  }

  .header-badge {
    background: #dbeafe;
    color: #1e40af;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 8pt;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  h1 {
    color: #0f172a;
    font-size: 16.5pt;
    margin: 2px 0 6px 0;
    font-weight: 700;
  }

  .meta-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 8.5pt;
    color: #475569;
    margin-top: 6px;
  }

  .meta-item {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    padding: 3px 9px;
    border-radius: 5px;
  }

  .meta-item strong {
    color: #0f172a;
  }

  h2 {
    color: #1e3a8a;
    font-size: 11.5pt;
    margin-top: 14px;
    margin-bottom: 6px;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    page-break-after: avoid;
  }

  h3 {
    color: #1e293b;
    font-size: 10pt;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }

  p {
    margin: 5px 0;
    text-align: justify;
  }

  ul, ol {
    margin: 5px 0;
    padding-left: 18px;
  }

  li {
    margin-bottom: 3px;
  }

  .alert {
    border-radius: 6px;
    padding: 9px 12px;
    margin: 9px 0;
    font-size: 8.8pt;
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

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 8pt;
    page-break-inside: avoid;
  }

  th {
    background-color: #f1f5f9;
    color: #1e293b;
    font-weight: 700;
    text-align: left;
    padding: 5px 7px;
    border: 1px solid #cbd5e1;
  }

  td {
    padding: 4.5px 7px;
    border: 1px solid #e2e8f0;
    vertical-align: middle;
  }

  tr:nth-child(even) {
    background-color: #f8fafc;
  }

  .badge-passed {
    background-color: #dcfce7;
    color: #166534;
    padding: 1.5px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 7.5pt;
  }

  .badge-warn {
    background-color: #fef3c7;
    color: #92400e;
    padding: 1.5px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 7.5pt;
  }

  .badge-danger {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 1.5px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 7.5pt;
  }

  pre, code {
    font-family: 'Consolas', 'Courier New', monospace;
  }

  pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 8px 10px;
    border-radius: 5px;
    font-size: 7.8pt;
    margin: 6px 0;
    line-height: 1.35;
    page-break-inside: avoid;
  }

  .code-inline {
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8pt;
    border: 1px solid #e2e8f0;
  }

  .footer-note {
    margin-top: 16px;
    border-top: 1px solid #cbd5e1;
    padding-top: 6px;
    font-size: 7.8pt;
    color: #64748b;
    display: flex;
    justify-content: space-between;
    page-break-inside: avoid;
  }
</style>
</head>
<body>

<div class="header-card">
  <span class="header-badge">CSE438 • Project 12 • Audited Phase 1 Technical Report</span>
  <h1>FractureFuseNet: Phase 1 Audit, Preprocessing & Multimodal Pipeline</h1>
  <div class="meta-grid">
    <div class="meta-item"><strong>Subtitle:</strong> Multi-Modal Late Fusion of Deep Radiographic Features & Classical Edge-Based Fracture-Line Detection</div>
    <div class="meta-item"><strong>Dataset:</strong> HBFMID (1,539 Audited / 1,429 Clean Included)</div>
    <div class="meta-item"><strong>Status:</strong> 12/12 Automated Checks Passed</div>
  </div>
</div>

<h2>1. Project Overview (প্রকল্পের মূল লক্ষ্য)</h2>
<p>
<strong>FractureFuseNet</strong> investigates a hybrid dual-branch architecture for 10-class bone fracture classification that fuses: (1) <strong>Branch A (Deep Radiographic Stream)</strong>—convolutional representations extracted from resized (224 × 224) radiographs capturing global anatomical structure, and (2) <strong>Branch B (Classical Edge/Line Stream)</strong>—a standardized 64-dimensional descriptor extracted via CLAHE, multi-threshold Canny edge detection, Probabilistic Hough Transform line geometry, and Sobel gradient orientation statistics.
</p>
<p>
<strong>বাংলা সারসংক্ষেপ:</strong> সাধারণ ডিপ লার্নিং (CNN) মডেল ছবির সামগ্রিক গঠন ভালো বুঝলেও অনেক সময় সূক্ষ্ম ফাটল (hairline fracture) মিস করে। তাই আমাদের <strong>FractureFuseNet</strong> মডেলে ডিপ ফিচারের সাথে ক্ল্যাসিক্যাল ইমেজ প্রসেসিং (Canny Edge ও Hough Line)-এর ৬৪টি জ্যামিতিক ফিচার যুক্ত করা হয়েছে। এই Phase 1-এ ডেটাসেটের সম্পূর্ণ অডিট, অ্যানোটেশন ক্লিনিং, ডুপ্লিকেট ও ডেটা লিকেজ সমস্যার সমাধান এবং পাইটর্চ মাল্টি-মোডাল পাইপলাইন যাচাই সম্পন্ন করা হয়েছে।
</p>

<h2>2. Dataset Description</h2>
<ul>
  <li><strong>Source Dataset:</strong> Human Bone Fractures Multi-modal Image Dataset (HBFMID, Mendeley Data <span class="code-inline">xwfs6xbk47</span>, YOLOv8 format).</li>
  <li><strong>Total Audited Records:</strong> <strong>1,539</strong> images (640 × 640 RGB) and <strong>1,539</strong> YOLO <span class="code-inline">.txt</span> annotation files.</li>
  <li><strong>Original Roboflow Partition:</strong> <span class="code-inline">train</span> = 1,347 (3× offline augmented from 449 raw images), <span class="code-inline">valid</span> = 128, <span class="code-inline">test</span> = 64.</li>
  <li><strong>Imaging Modalities:</strong> <strong>X-ray (<span class="code-inline">xray</span>):</strong> 1,337 images (86.87%) | <strong>MRI (<span class="code-inline">mri</span>):</strong> 202 images (13.13%, marked by <span class="code-inline">_mri_</span> in filenames).</li>
  <li><strong>10 Target Classes:</strong> <span class="code-inline">Comminuted (0)</span>, <span class="code-inline">Greenstick (1)</span>, <span class="code-inline">Healthy (2)</span>, <span class="code-inline">Linear (3)</span>, <span class="code-inline">Oblique Displaced (4)</span>, <span class="code-inline">Oblique (5)</span>, <span class="code-inline">Segmental (6)</span>, <span class="code-inline">Spiral (7)</span>, <span class="code-inline">Transverse Displaced (8)</span>, <span class="code-inline">Transverse (9)</span>.</li>
</ul>

<h2>3. Dataset Auditing (ডেটা অডিট ও গুণগত মান যাচাই)</h2>
<table>
  <thead>
    <tr>
      <th>Audit Metric</th>
      <th>Measured Result</th>
      <th>Status</th>
      <th>Verification Detail</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>Total Image & Label Files</strong></td><td>1,539 images / 1,539 labels</td><td><span class="badge-passed">VERIFIED</span></td><td>100% 1-to-1 correspondence on disk</td></tr>
    <tr><td><strong>Corrupt / Unreadable Files</strong></td><td>0</td><td><span class="badge-passed">PASSED</span></td><td>All 1,539 images verified via PIL & OpenCV</td></tr>
    <tr><td><strong>Image Resolution & Channels</strong></td><td>640 × 640 RGB (100%)</td><td><span class="badge-passed">UNIFORM</span></td><td>3-channel grayscale radiographs</td></tr>
    <tr><td><strong>YOLO Bounding Boxes</strong></td><td>1,771 valid boxes</td><td><span class="badge-passed">PASSED</span></td><td>0 invalid class IDs; 0 out-of-bounds coordinates</td></tr>
    <tr><td><strong>Exact MD5 Byte Duplicates</strong></td><td>0</td><td><span class="badge-passed">PASSED</span></td><td>No byte-identical files across splits</td></tr>
  </tbody>
</table>

<h2>4. Annotation Cleaning & Inclusion Policy (অ্যানোটেশন ক্লিনিং নীতি)</h2>
<p>
Across 1,539 records, 220 images contain multiple boxes and 113 contain multiple classes. Instead of silently picking the first bounding box on tied multi-class records, every image was categorized into four <span class="code-inline">label_quality</span> tiers:
</p>
<table>
  <thead>
    <tr>
      <th><span class="code-inline">label_quality</span> Tier</th>
      <th>Count</th>
      <th>Share</th>
      <th>Policy Decision & Rationale (বাংলা ব্যাখ্যা সহ)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong><span class="code-inline">single</span></strong></td><td><strong>1,423</strong></td><td>92.46%</td><td><span class="badge-passed">INCLUDED</span> Single unambiguous class across all boxes in the image.</td></tr>
    <tr><td><strong><span class="code-inline">majority</span></strong></td><td><strong>6</strong></td><td>0.39%</td><td><span class="badge-passed">INCLUDED</span> Multi-class with strict majority (<span class="code-inline">52_jpg.rf.*</span>: 2 boxes Transverse Displaced, 1 Oblique Displaced).</td></tr>
    <tr><td><strong><span class="code-inline">tie</span></strong></td><td><strong>107</strong></td><td>6.95%</td><td><span class="badge-warn">EXCLUDED</span> Exact box-count tie across classes (<span class="code-inline">Ambiguous_Tie</span>). ভুল লেবেলিং এড়াতে সিঙ্গেল-লেবেল স্প্লিট থেকে বাদ দেওয়া হয়েছে।</td></tr>
    <tr><td><strong><span class="code-inline">unannotated</span></strong></td><td><strong>3</strong></td><td>0.19%</td><td><span class="badge-danger">EXCLUDED</span> 0-byte empty label files (<span class="code-inline">114_jpg.rf.*</span> in train).</td></tr>
    <tr><td><strong>Total Audited</strong></td><td><strong>1,539</strong></td><td>100.00%</td><td><strong>1,429 Included Clean Records (92.85%) | 110 Excluded Records (7.15%, logged in audit CSV)</strong></td></tr>
  </tbody>
</table>

<h2>5. Duplicate Detection & Source Grouping Analysis</h2>
<ul>
  <li><strong>Filename Prefix (<span class="code-inline">base_id</span>):</strong> <span class="code-inline">filename.split('.rf.')[0]</span> yields <strong>245</strong> unique <span class="code-inline">base_id</span> prefixes across 1,539 images (238 in the 1,429 included images). However, <strong>158 of 245 <span class="code-inline">base_id</span>s (64.49%)</strong> contain more than one distinct source radiograph because Roboflow flattened separate class folders numbered <span class="code-inline">1.jpg..225.jpg</span>.</li>
  <li><strong>Dihedral Perceptual Hash Clustering (<span class="code-inline">dup_group_id</span>):</strong> Computing 64-bit <span class="code-inline">pHash</span> + 64-bit <span class="code-inline">dHash</span> across all 4 dihedral orientations (0°, H-flip, V-flip, 180°) and clustering at Hamming distance ≤ 10 bits identifies <strong>657</strong> perceptual duplicate clusters (612 in the 1,429 included images), matching the ~641 raw source images reported by Mendeley.</li>
  <li><strong>Composite Grouping (<span class="code-inline">group_id</span>):</strong> Union-Find uniting <span class="code-inline">base_id</span> and <span class="code-inline">dup_group_id</span> yields <strong>242</strong> composite groups overall (<strong>235</strong> across the 1,429 included records), ensuring neither <span class="code-inline">base_id</span> nor perceptual duplicates ever cross splits.</li>
</ul>

<h2>6. Data Leakage Investigation in the Original Roboflow Split</h2>
<div class="alert alert-warning">
  <strong>Critical Leakage in Original Roboflow Split:</strong>
  In the original partition (1,347 train / 128 valid / 64 test), <strong>100 of 245 <span class="code-inline">base_id</span>s (40.82%)</strong> and <strong>84 of 657 perceptual duplicate clusters (12.79%)</strong> span multiple splits (48 train+valid, 31 train+test, 2 valid+test, 3 train+valid+test). Furthermore, the original 64-image test split contains <strong>0 <span class="code-inline">Linear</span></strong> and <strong>0 <span class="code-inline">Segmental</span></strong> images.
</div>

<h2>7. Corrected Splitting Methodology (<span class="code-inline">stratified_group_split</span>)</h2>
<p>
Using 5-fold <span class="code-inline">StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)</span> on the 1,429 included records grouped by composite <span class="code-inline">group_id</span> (preserving original DataFrame indices):
</p>
<ul>
  <li><strong>Train (<span class="code-inline">cv_fold</span> ∈ {0, 2, 4}):</strong> <strong>857 images (59.97%)</strong> | 143 <span class="code-inline">base_id</span>s | 362 <span class="code-inline">dup_group_id</span>s | 141 composite <span class="code-inline">group_id</span>s | 10/10 classes.</li>
  <li><strong>Valid (<span class="code-inline">cv_fold</span> = 1):</strong> <strong>285 images (19.94%)</strong> | 47 <span class="code-inline">base_id</span>s | 126 <span class="code-inline">dup_group_id</span>s | 46 composite <span class="code-inline">group_id</span>s | 10/10 classes.</li>
  <li><strong>Test (<span class="code-inline">cv_fold</span> = 3):</strong> <strong>287 images (20.08%)</strong> | 48 <span class="code-inline">base_id</span>s | 124 <span class="code-inline">dup_group_id</span>s | 48 composite <span class="code-inline">group_id</span>s | 9/10 classes.</li>
  <li><strong>Automated Zero-Leakage Assertions:</strong> 0 <span class="code-inline">base_id</span> overlap, 0 <span class="code-inline">dup_group_id</span> overlap, 0 <span class="code-inline">group_id</span> overlap, and 0 excluded records in active splits.</li>
</ul>

<h2>8. Class & Modality Distribution Across Splits</h2>
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Class Name</th>
      <th>Train Imgs (Groups)</th>
      <th>Valid Imgs (Groups)</th>
      <th>Test Imgs (Groups)</th>
      <th>Total Clean</th>
      <th>X-ray Imgs</th>
      <th>MRI Imgs (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td><strong>Comminuted</strong></td><td>94 (30)</td><td>32 (12)</td><td>32 (12)</td><td><strong>158</strong></td><td>56</td><td><strong>102 (64.6%)</strong></td></tr>
    <tr><td>1</td><td><strong>Greenstick</strong></td><td>50 (8)</td><td>16 (1)</td><td>17 (3)</td><td><strong>83</strong></td><td>77</td><td>6 (7.2%)</td></tr>
    <tr><td>2</td><td><strong>Healthy</strong></td><td>41 (6)</td><td>12 (2)</td><td>11 (2)</td><td><strong>64</strong></td><td>64</td><td>0 (0.0%)</td></tr>
    <tr><td>3</td><td><strong>Linear</strong></td><td>12 (3)</td><td>3 (1)</td><td>7 (1)</td><td><strong>22</strong></td><td>19</td><td>3 (13.6%)</td></tr>
    <tr><td>4</td><td><strong>Oblique Displaced</strong></td><td>185 (30)</td><td>61 (10)</td><td>61 (10)</td><td><strong>307</strong></td><td>282</td><td>25 (8.1%)</td></tr>
    <tr><td>5</td><td><strong>Oblique</strong></td><td>31 (5)</td><td>9 (1)</td><td>12 (2)</td><td><strong>52</strong></td><td>52</td><td>0 (0.0%)</td></tr>
    <tr><td>6</td><td><strong>Segmental*</strong></td><td>5 (1)</td><td>3 (1)</td><td><strong>0* (0)</strong></td><td><strong>8</strong></td><td>8</td><td>0 (0.0%)</td></tr>
    <tr><td>7</td><td><strong>Spiral</strong></td><td>43 (7)</td><td>16 (3)</td><td>15 (2)</td><td><strong>74</strong></td><td>74</td><td>0 (0.0%)</td></tr>
    <tr><td>8</td><td><strong>Transverse Displaced</strong></td><td>340 (58)</td><td>114 (19)</td><td>114 (19)</td><td><strong>568</strong></td><td>502</td><td>66 (11.6%)</td></tr>
    <tr><td>9</td><td><strong>Transverse</strong></td><td>56 (9)</td><td>19 (3)</td><td>18 (3)</td><td><strong>93</strong></td><td>93</td><td>0 (0.0%)</td></tr>
    <tr><td>—</td><td><strong>Total Included</strong></td><td><strong>857 (141)</strong></td><td><strong>285 (46)</strong></td><td><strong>287 (48)</strong></td><td><strong>1,429</strong></td><td><strong>1,227</strong></td><td><strong>202 (14.1%)</strong></td></tr>
  </tbody>
</table>
<p style="font-size: 7.8pt; color: #475569;">
*<strong>Mathematically Unavoidable Constraint on <span class="code-inline">Segmental</span>:</strong> In the entire cleaned dataset, <span class="code-inline">Segmental</span> has only 8 single-label images belonging to <strong>2 <span class="code-inline">base_id</span> groups</strong> (<span class="code-inline">70_jpg</span> with 5 images in Train; <span class="code-inline">225_jpg</span> with 3 images in Valid). By the Pigeonhole Principle, 2 disjoint groups cannot populate 3 disjoint splits without data leakage. Both groups are evaluated in 5-fold cross-validation via <span class="code-inline">cv_fold</span>.
</p>

<h2>9. Classical CV Feature Extraction & Dataset-Wide Statistical Verification</h2>
<p>
We evaluated the corrected 64-dimensional classical feature vector (with axial circular statistics <span class="code-inline">2θ</span>, <span class="code-inline">log1p</span> line count, Shannon orientation entropy, and zero-gradient protection; brightness/contrast shortcuts removed) across all <strong>1,429 included radiographs</strong> and within <strong>X-ray only (N = 1,227: 64 Healthy vs. 1,163 Fracture)</strong>:
</p>
<table>
  <thead>
    <tr>
      <th>Classical CV Metric (Unscaled Physical Units)</th>
      <th>Healthy X-ray (n=64) Mean ± SD / Med (IQR)</th>
      <th>Fracture X-ray (n=1,163) Mean ± SD / Med (IQR)</th>
      <th>Mann-Whitney U p-value (X-ray)</th>
      <th>10-Class Kruskal-Wallis H (p-value, ε²)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>High-Threshold Canny Density (%)</strong></td>
      <td>3.00±0.88 / 2.96 (0.73)</td>
      <td>2.06±1.24 / 1.88 (1.28)</td>
      <td>6.89 × 10⁻¹⁷ (r = -0.619)</td>
      <td>H = 252.16 (p = 3.49 × 10⁻⁴⁹, ε² = 0.177)</td>
    </tr>
    <tr>
      <td><strong>Low-Threshold Canny Density (%)</strong></td>
      <td>6.96±1.54 / 6.87 (2.00)</td>
      <td>5.22±2.91 / 4.76 (2.97)</td>
      <td>3.07 × 10⁻¹⁵ (r = -0.585)</td>
      <td>H = 195.26 (p = 3.26 × 10⁻³⁷, ε² = 0.137)</td>
    </tr>
    <tr>
      <td><strong>Hough Line Segment Count</strong></td>
      <td>199.5±110.0 / 181.5 (154.5)</td>
      <td>109.3±117.2 / 79.0 (100.0)</td>
      <td>7.83 × 10⁻¹⁴ (r = -0.554)</td>
      <td>H = 239.68 (p = 1.50 × 10⁻⁴⁶, ε² = 0.168)</td>
    </tr>
    <tr>
      <td><strong>Average Hough Line Length (px)</strong></td>
      <td>51.51±11.41 / 47.53 (11.39)</td>
      <td><strong>61.75±20.69 / 56.33 (22.09)</strong></td>
      <td><strong>3.60 × 10⁻⁸ (r = +0.409)</strong></td>
      <td>H = 192.55 (p = 1.20 × 10⁻³⁶, ε² = 0.135)</td>
    </tr>
    <tr>
      <td><strong>Mean Sobel Gradient Magnitude</strong></td>
      <td>31.72±5.59 / 31.43 (8.17)</td>
      <td>26.68±8.95 / 25.22 (8.76)</td>
      <td>3.09 × 10⁻¹² (r = -0.517)</td>
      <td>H = 213.06 (p = 6.03 × 10⁻⁴¹, ε² = 0.149)</td>
    </tr>
    <tr>
      <td><strong>Gradient Orientation Entropy</strong></td>
      <td>0.955±0.025 / 0.962 (0.032)</td>
      <td>0.946±0.034 / 0.954 (0.044)</td>
      <td>4.49 × 10⁻² (r = -0.148)</td>
      <td>H = 54.45 (p = 1.55 × 10⁻⁸, ε² = 0.038)</td>
    </tr>
  </tbody>
</table>
<div class="alert alert-info">
  <strong>Scientific Finding:</strong> Classical Hough lines detect both normal cortical shafts and fracture discontinuities. While Healthy multi-bone joint views have higher total edge density (2.96% vs 1.88%), fractured long-bone radiographs exhibit significantly longer linear segments (median 56.33 px vs 47.53 px, p = 3.60 × 10⁻⁸) and strong 10-class statistical separability (Kruskal-Wallis ε² up to 0.177, p &lt; 10⁻³⁶).
</div>

<h2>10. Multimodal Preprocessing & Shared Geometric Augmentation (<span class="code-inline">src/dataset.py</span>)</h2>
<pre>
[ Raw Radiograph (640x640 RGB) ]
       │
       ▼
[ Shared Geometric Augmentation: apply_shared_geometric_transform() ]
  * Train: RandomHorizontalFlip (p=0.5) + RandomRotation (±15°, bilinear) applied ONCE
  * Valid / Test: Deterministic Identity (no random geometric distortion)
       │
       ├──► Branch A (Deep Vision): Resize(224x224) -> ToTensor() -> Normalize(Train Mean=0.2555, Train Std=0.2594)
       │    Output Tensor: torch.float32 [Batch, 3, 224, 224]
       │
       └──► Branch B (Classical CV): Grayscale(640x640) -> CLAHE -> Canny + HoughLinesP + Sobel -> 64-dim vector
            -> Train-Fitted StandardScaler (N=857 train images, edge_feature_scaler.joblib)
            Output Tensor: torch.float32 [Batch, 64]
</pre>

<h2>11. Master Pipeline Verification Results (<span class="code-inline">run_phase1_pipeline.py</span>)</h2>
<ul>
  <li><strong>Schema & File Integrity:</strong> 1,539 rows, 17 standardized columns, 0 missing files, 0 backslash paths.</li>
  <li><strong>Zero Leakage & Clean Split Assertions:</strong> 0 <span class="code-inline">base_id</span>, 0 <span class="code-inline">dup_group_id</span>, and 0 <span class="code-inline">group_id</span> overlaps; 0 ties/empty labels in active splits.</li>
  <li><strong>Shared Geometric Symmetry & Scaler Audit:</strong> Exact horizontal flip symmetry verified; <span class="code-inline">StandardScaler</span> verified with <span class="code-inline">n_samples_seen_ == 857</span>.</li>
  <li><strong>DataLoader Batch Verification (Batch Size = 16):</strong> <span class="code-inline">TRAIN</span> (54 batches), <span class="code-inline">VALID</span> (18 batches), and <span class="code-inline">TEST</span> (18 batches) verified with exact shapes <span class="code-inline">(16, 3, 224, 224)</span>, <span class="code-inline">(16, 64)</span>, <span class="code-inline">(16,)</span> and 0 NaN/Inf values.</li>
</ul>

<h2>12. Limitations & Methodological Caveats</h2>
<ul>
  <li><strong>Rare Class Sample Sizes:</strong> <span class="code-inline">Segmental</span> (8 clean images across 2 <span class="code-inline">base_id</span>s) has 0 test images in the fixed split due to the non-overlapping group constraint, and <span class="code-inline">Linear</span> has 22 clean images across 5 groups. Phase 2 will report both 9-class fixed-test macro-F1 and 5-fold cross-validated metrics across <span class="code-inline">cv_fold</span>.</li>
  <li><strong>Modality Confounding:</strong> MRI images (13.13% overall) represent 64.56% of <span class="code-inline">Comminuted</span> fractures while 6 classes have 0% MRI. Phase 2 must evaluate overall and X-ray-only subsets separately.</li>
</ul>

<h2>13. Phase 2 Roadmap</h2>
<p>
Phase 2 will benchmark: (1) <strong>Deep-Only CNN Baselines</strong> (EfficientNet-B0 / ResNet-50), (2) <strong>Classical-Only ML Baselines</strong> (Random Forest / XGBoost / MLP on the 64-dim vector), and (3) <strong>FractureFuseNet Late Fusion</strong> with class-weighted Cross-Entropy / Focal Loss and modality-stratified ablation studies.
</p>

<div class="footer-note">
  <span>Course CSE438 • Project 12: FractureFuseNet (Audited Phase 1 Report)</span>
  <span>Human Bone Fractures Multi-modal Image Dataset (HBFMID)</span>
</div>

</body>
</html>
"""

html_path = str(PHASE1_DIR / "PHASE_1_SUMMARY.html")
pdf_path = str(PHASE1_DIR / "PHASE_1_SUMMARY.pdf")

with open(html_path, "w", encoding="utf-8") as f:
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

print("Regenerating publication-quality PDF via Microsoft Edge headless...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)

if os.path.exists(pdf_path):
    print(f"SUCCESS: Clean PDF generated at: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path):,} bytes")
else:
    raise RuntimeError(f"Failed to generate PDF: {res.stderr}")
