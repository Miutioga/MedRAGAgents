const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

// ============================================================
// PRESENTATION CONFIGURATION & THEME SYSTEM FOR CANVA IMPORT
// ============================================================
pres.layout = "LAYOUT_16x9"; // 10.0 x 5.625 inches
pres.title = "MedRAGAgents Midterm Presentation";
pres.subject = "Medical Multi-Agent LLM System for Factual Accuracy on MedQA-USMLE";
pres.author = "xcentralnn (centralngx)";

// Canva-optimized Color System
const C = {
  BG_DARK:     "0F172A", // Deep Slate Navy
  CARD_BG:     "1E293B", // Card Fill
  CARD_BORDER: "334155", // Card Border Stroke
  CODE_BG:     "090D16", // Monospace Code Fill
  TEXT_MAIN:   "F8FAFC", // White Main Text
  TEXT_MUTED:  "94A3B8", // Muted Gray Text
  CYAN:        "0EA5E9", // Accent Cyan
  EMERALD:     "10B981", // Accent Emerald / Success
  AMBER:       "F59E0B", // Accent Amber / Highlight
  CRIMSON:     "EF4444", // Accent Crimson / Challenge
  PURPLE:      "8B5CF6", // Accent Purple / Memory
  WHITE:       "FFFFFF",
};

const FONT_MAIN = "Segoe UI";
const FONT_CODE = "Consolas";

// Define Master Slide Template for Canva Import
pres.defineSlideMaster({
  title: "CANVA_MASTER",
  background: { color: C.BG_DARK },
  objects: [
    { rect: { x: 0, y: 0, w: "100%", h: 0.08, fill: { color: C.CYAN } } },
    { rect: { x: 0.6, y: 5.25, w: 8.8, h: 0.02, fill: { color: C.CARD_BORDER } } },
    {
      text: {
        text: "MedRAGAgents | Midterm Presentation - Machine Learning in Security (UIT-SDH)",
        options: { x: 0.6, y: 5.3, w: 6.0, h: 0.25, fontFace: FONT_MAIN, fontSize: 9, color: C.TEXT_MUTED }
      }
    }
  ]
});

// Helper: Add Standard Header
function addSlideHeader(slide, titleText, categoryText) {
  slide.addText(categoryText.toUpperCase(), {
    x: 0.6, y: 0.35, w: 8.8, h: 0.25,
    fontFace: FONT_MAIN, fontSize: 10, bold: true, color: C.CYAN, charSpacing: 1.5
  });
  slide.addText(titleText, {
    x: 0.6, y: 0.6, w: 8.8, h: 0.5,
    fontFace: FONT_MAIN, fontSize: 20, bold: true, color: C.TEXT_MAIN
  });
}

// ============================================================
// SLIDE 1: TITLE SLIDE
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 0.8, w: 8.8, h: 4.2,
    fill: { color: C.CARD_BG }, line: { color: C.CYAN, width: 1.5 }, rectRadius: 0.1
  });

  slide.addText("DE TAI GIUA KY — MACHINE LEARNING IN SECURITY (UIT-SDH)", {
    x: 0.9, y: 1.1, w: 8.2, h: 0.3,
    fontFace: FONT_MAIN, fontSize: 11, bold: true, color: C.CYAN
  });

  slide.addText("Medical Multi-Agent LLM System for\nFactual Accuracy on MedQA-USMLE", {
    x: 0.9, y: 1.5, w: 8.2, h: 1.2,
    fontFace: FONT_MAIN, fontSize: 26, bold: true, color: C.TEXT_MAIN, lineSpacing: 32
  });

  slide.addText("He Thong Da Agent Chuyen Gia Y Khoa, Truy Xuat Chung Cu MedRAG & Bo Nho Hai Tang (Dual-Layer Memory)", {
    x: 0.9, y: 2.8, w: 8.2, h: 0.6,
    fontFace: FONT_MAIN, fontSize: 13, color: C.TEXT_MUTED
  });

  const badges = [
    { label: "Benchmark: MedQA (1,273 test)", color: C.EMERALD },
    { label: "LLM Core: Gemini 3.6 Flash", color: C.PURPLE },
    { label: "GitHub: xcentralnn/MedRAGAgents", color: C.AMBER }
  ];
  badges.forEach((b, i) => {
    slide.addShape(pres.ShapeType.roundRect, {
      x: 0.9 + i * 2.7, y: 3.6, w: 2.5, h: 0.4,
      fill: { color: C.BG_DARK }, line: { color: b.color, width: 1 }, rectRadius: 0.05
    });
    slide.addText(b.label, {
      x: 0.9 + i * 2.7, y: 3.6, w: 2.5, h: 0.4,
      fontFace: FONT_MAIN, fontSize: 9.5, bold: true, color: b.color, align: "center", valign: "middle"
    });
  });
}

// ============================================================
// SLIDE 2: MOTIVATION & TASK FORMULATION
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Dat Van De & Rang Buoc Dau Vao / Dau Ra MedQA", "01. MOTIVATION & TASK FORMULATION");

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.2, w: 4.2, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.CRIMSON, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Thach Thuc Lap Luan Y Khoa", {
    x: 0.8, y: 1.4, w: 3.8, h: 0.3,
    fontFace: FONT_MAIN, fontSize: 14, bold: true, color: C.CRIMSON
  });
  slide.addText([
    { text: "• Ao giac LLM (Hallucination): ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "LLM don le de tu tin dua ra chan doan sai khi gap ca benh phuc tap dan xen nhieu chuyen khoa.\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Thieu tri thuc cap nhat: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Du lieu huan luyen LLM tinh, thieu truy xuat van ban y khoa chinh thong (StatPearls, Textbooks).\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Thieu co che kiem dinh: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Khong co phan bien da chieu (Cross-examination) dan den sai sot day chuyen.", options: { color: C.TEXT_MUTED } }
  ], { x: 0.8, y: 1.8, w: 3.8, h: 3.0, fontFace: FONT_MAIN, fontSize: 10.5, lineSpacing: 15 });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 5.0, y: 1.2, w: 4.4, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.CYAN, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Dinh Dang Dau Vao / Dau Ra Rang Buoc", {
    x: 5.2, y: 1.4, w: 4.0, h: 0.3,
    fontFace: FONT_MAIN, fontSize: 14, bold: true, color: C.CYAN
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 5.2, y: 1.8, w: 4.0, h: 1.3,
    fill: { color: C.CODE_BG }, line: { color: C.CARD_BORDER, width: 1 }, rectRadius: 0.05
  });
  slide.addText("INPUT SPECIFICATION (MedQA)", {
    x: 5.3, y: 1.9, w: 3.8, h: 0.2, fontFace: FONT_CODE, fontSize: 9, bold: true, color: C.AMBER
  });
  slide.addText("• Question: Clinical case vignette string\n• Options: { 'A': '...', 'B': '...', 'C': '...', 'D': '...' }\n• Ground Truth: gold_answer ('A'/'B'/'C'/'D')", {
    x: 5.3, y: 2.15, w: 3.8, h: 0.9, fontFace: FONT_CODE, fontSize: 9.5, color: C.TEXT_MAIN
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 5.2, y: 3.2, w: 4.0, h: 1.6,
    fill: { color: C.CODE_BG }, line: { color: C.EMERALD, width: 1 }, rectRadius: 0.05
  });
  slide.addText("OUTPUT REQUIREMENT (STRICT)", {
    x: 5.3, y: 3.3, w: 3.8, h: 0.2, fontFace: FONT_CODE, fontSize: 9, bold: true, color: C.EMERALD
  });
  slide.addText("• pred_answer: Dung 1 ky tu lua chon ('A'/'B'/'C'/'D')\n• raw_output: Bao cao phan tich & loi giai thich\n• Constraint: Invalid Response Rate giu o muc 0.0%", {
    x: 5.3, y: 3.55, w: 3.8, h: 1.1, fontFace: FONT_CODE, fontSize: 9.5, color: C.TEXT_MAIN
  });
}

// ============================================================
// SLIDE 3: SYSTEM ARCHITECTURE & AGENT PIPELINE
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Kien Truc Multi-Agent Da Chuyen Gia (5 Agents Core)", "02. SYSTEM ARCHITECTURE & AGENT WORKFLOW");

  const agents = [
    { title: "1. Classifier", code: "DomainAgent", desc: "Phan loai cau hoi & phuong an chon vao 5 chuyên khoa sau (Pathology, Pharmacology...)." },
    { title: "2. Analysis", code: "AnalysisAgent", desc: "Kich hoat cac Agent chuyen gia phan tich ca benh doc lap theo goc nhin tung nganh." },
    { title: "3. MedRAG", code: "RAGAgent", desc: "Truy xuat chung cu y khoa tu Textbooks/PubMed bang MedCPT vector (Top-32)." },
    { title: "4. Synthesis", code: "SynthesisAgent", desc: "Tong hop cac bai phan tich chuyen gia & tri thuc RAG thanh bao cao hop nhat." },
    { title: "5. Verifier", code: "VerifierAgent", desc: "Bo phieu & sua doi dong thuan (toi da 3 rounds) nham loai bo mau thuan." }
  ];

  agents.forEach((ag, idx) => {
    let posX = 0.6 + idx * 1.78;
    slide.addShape(pres.ShapeType.roundRect, {
      x: posX, y: 1.3, w: 1.68, h: 3.7,
      fill: { color: C.CARD_BG }, line: { color: idx === 4 ? C.AMBER : C.CYAN, width: 1 }, rectRadius: 0.08
    });
    slide.addText(ag.title, {
      x: posX + 0.05, y: 1.5, w: 1.58, h: 0.45, fontFace: FONT_MAIN, fontSize: 11, bold: true, color: C.TEXT_MAIN, align: "center"
    });
    slide.addShape(pres.ShapeType.roundRect, {
      x: posX + 0.1, y: 2.1, w: 1.48, h: 0.25, fill: { color: C.CODE_BG }, rectRadius: 0.03
    });
    slide.addText(ag.code, {
      x: posX + 0.1, y: 2.1, w: 1.48, h: 0.25, fontFace: FONT_CODE, fontSize: 9, color: C.CYAN, align: "center", valign: "middle"
    });
    slide.addText(ag.desc, {
      x: posX + 0.1, y: 2.5, w: 1.48, h: 2.3, fontFace: FONT_MAIN, fontSize: 9.5, color: C.TEXT_MUTED, lineSpacing: 13
    });
  });
}

// ============================================================
// SLIDE 4: DUAL-LAYER MEMORY SYSTEM
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Co Che Bo Nho Hai Tang (Short-Term & Long-Term Memory)", "03. DUAL-LAYER MEMORY SYSTEM");

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.2, w: 4.2, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.PURPLE, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Short-Term Memory (In-RAM State)", {
    x: 0.8, y: 1.4, w: 3.8, h: 0.3, fontFace: FONT_MAIN, fontSize: 14, bold: true, color: C.PURPLE
  });
  slide.addText([
    { text: "• Scope: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Ton tai trong RAM trong suot luot xu ly cua 1 cau hoi.\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Vai tro: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Luu giu trang thai trung gian giua cac Agent (domain tags, analyses, RAG snippets, vote history).\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Lifecycle: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Tu dong reset (`reset_short()`) khi chuyen cau hoi moi de tranh nhiem doc ngu canh.", options: { color: C.TEXT_MUTED } }
  ], { x: 0.8, y: 1.8, w: 3.8, h: 3.0, fontFace: FONT_MAIN, fontSize: 10.5, lineSpacing: 15 });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 5.0, y: 1.2, w: 4.4, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.EMERALD, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Long-Term Memory (Persistent Cache)", {
    x: 5.2, y: 1.4, w: 4.0, h: 0.3, fontFace: FONT_MAIN, fontSize: 14, bold: true, color: C.EMERALD
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 5.2, y: 1.8, w: 4.0, h: 1.5, fill: { color: C.CODE_BG }, line: { color: C.CARD_BORDER, width: 1 }, rectRadius: 0.05
  });
  slide.addText("SHA-256 DISK CACHE KEYING (memory.py)", {
    x: 5.3, y: 1.9, w: 3.8, h: 0.2, fontFace: FONT_CODE, fontSize: 9, bold: true, color: C.AMBER
  });
  slide.addText("def make_key(question: str, options: Any) -> str:\n    raw = question.strip() + str(options)\n    return hashlib.sha256(\n        raw.encode('utf-8')\n    ).hexdigest()[:16]", {
    x: 5.3, y: 2.15, w: 3.8, h: 1.0, fontFace: FONT_CODE, fontSize: 9.5, color: C.TEXT_MAIN
  });

  slide.addText([
    { text: "• Toi uu Latency & Cost: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Bo qua suy luan LLM cho cac cau hoi trung lap (from_cache=True).\n", options: { color: C.TEXT_MUTED } },
    { text: "• Do tre 0ms: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Chi phi 0 token cho cac cau hoi da luu trong file JSON disk cache.", options: { color: C.TEXT_MUTED } }
  ], { x: 5.2, y: 3.4, w: 4.0, h: 1.4, fontFace: FONT_MAIN, fontSize: 10.5, lineSpacing: 14 });
}

// ============================================================
// SLIDE 5: SYSTEM VARIANTS MATRIX (V0 TO V3)
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Ma Tran 4 Bien The Thu Nghiem (V0 -> V3)", "04. EXPERIMENTAL VARIANTS MATRIX");

  const tableData = [
    [
      { text: "Bien The (Variant)", options: { bold: true, fill: C.CYAN, color: C.WHITE, fontFace: FONT_MAIN, fontSize: 10 } },
      { text: "Direct CoT LLM", options: { bold: true, fill: C.CYAN, color: C.WHITE, fontFace: FONT_MAIN, fontSize: 10 } },
      { text: "RAG Retrieval", options: { bold: true, fill: C.CYAN, color: C.WHITE, fontFace: FONT_MAIN, fontSize: 10 } },
      { text: "Multi-Agent Workflow", options: { bold: true, fill: C.CYAN, color: C.WHITE, fontFace: FONT_MAIN, fontSize: 10 } },
      { text: "Consensus Verifier", options: { bold: true, fill: C.CYAN, color: C.WHITE, fontFace: FONT_MAIN, fontSize: 10 } },
      { text: "Dual Memory", options: { bold: true, fill: C.CYAN, color: C.WHITE, fontFace: FONT_MAIN, fontSize: 10 } }
    ],
    [
      { text: "V0 (Direct LLM)", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "✓ CoT Direct", options: { color: C.EMERALD } },
      { text: "✗ None", options: { color: C.CRIMSON } },
      { text: "✗ None", options: { color: C.CRIMSON } },
      { text: "✗ None", options: { color: C.CRIMSON } },
      { text: "✗ None", options: { color: C.CRIMSON } }
    ],
    [
      { text: "V1 (RAG-Only)", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "✓ Prompting", options: { color: C.EMERALD } },
      { text: "✓ MedCPT (Top-32)", options: { color: C.EMERALD } },
      { text: "✗ None", options: { color: C.CRIMSON } },
      { text: "✗ None", options: { color: C.CRIMSON } },
      { text: "✗ None", options: { color: C.CRIMSON } }
    ],
    [
      { text: "V2 (Multi-Agent)", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "✓ Multi-Prompt", options: { color: C.EMERALD } },
      { text: "✓ MedCPT (Top-32)", options: { color: C.EMERALD } },
      { text: "✓ 5 Expert Agents", options: { color: C.EMERALD } },
      { text: "✓ Max 3 Rounds", options: { color: C.EMERALD } },
      { text: "✗ Stateless", options: { color: C.CRIMSON } }
    ],
    [
      { text: "V3 (Full System)", options: { bold: true, color: C.AMBER } },
      { text: "✓ Multi-Prompt", options: { color: C.EMERALD } },
      { text: "✓ MedCPT (Top-32)", options: { color: C.EMERALD } },
      { text: "✓ 5 Expert Agents", options: { color: C.EMERALD } },
      { text: "✓ Max 3 Rounds", options: { color: C.EMERALD } },
      { text: "✓ Short + Long Term", options: { color: C.EMERALD } }
    ]
  ];

  slide.addTable(tableData, {
    x: 0.6, y: 1.3, w: 8.8,
    colW: [2.0, 1.35, 1.35, 1.4, 1.35, 1.35],
    border: { pt: 1, color: C.CARD_BORDER },
    fill: C.CARD_BG,
    fontFace: FONT_MAIN, fontSize: 10, color: C.TEXT_MUTED, align: "center", valign: "middle"
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 4.1, w: 8.8, h: 0.9,
    fill: { color: C.CARD_BG }, line: { color: C.CYAN, width: 1 }, rectRadius: 0.05
  });
  slide.addText("Y Nghia Danh Gia Ablation Study:", {
    x: 0.8, y: 4.2, w: 8.4, h: 0.25, fontFace: FONT_MAIN, fontSize: 11, bold: true, color: C.CYAN
  });
  slide.addText("Co lap chinh xac muc do dong gop (Accuracy Gain) cua tung thanh phan: RAG Retrieval (V1-V0), Multi-Agent Consensus (V2-V1), va Memory Caching Layer (V3-V2).", {
    x: 0.8, y: 4.45, w: 8.4, h: 0.45, fontFace: FONT_MAIN, fontSize: 10, color: C.TEXT_MUTED
  });
}

// ============================================================
// SLIDE 6: BENCHMARK PROTOCOL & STATISTICAL METRICS
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Quy Trinh & Cong Thuc Danh Gia Thong Ke (evaluate.py)", "05. BENCHMARK PROTOCOL & STATISTICAL METRICS");

  const cards = [
    {
      title: "1. Primary Accuracy Metric", color: C.CYAN,
      eq: "Accuracy = Correct / N",
      text: "• Quy mo test set chinh thuc: N = 1,273 cau MedQA-USMLE.\n• Accuracy Gain: Accuracy(V3) - Accuracy(V0)."
    },
    {
      title: "2. McNemar Significance Test", color: C.EMERALD,
      eq: "chi^2 = (|b - c| - 1)^2 / (b + c)",
      text: "• b: So cau V0 dung nhung V3 sai.\n• c: So cau V3 dung nhung V0 sai.\n• Xac nhan su cai thien co y nghia thong ke (p < 0.05)."
    },
    {
      title: "3. Bootstrap 95% CI & Error", color: C.AMBER,
      eq: "CI_95% = [Q_0.025, Q_0.975]",
      text: "• Lay mau lai Bootstrap (B = 1000 iterations).\n• Invalid Rate = Invalid Predictions / N (Muc tiêu 0%)."
    }
  ];

  cards.forEach((c, idx) => {
    let posX = 0.6 + idx * 2.98;
    slide.addShape(pres.ShapeType.roundRect, {
      x: posX, y: 1.3, w: 2.84, h: 3.7,
      fill: { color: C.CARD_BG }, line: { color: c.color, width: 1 }, rectRadius: 0.08
    });
    slide.addText(c.title, {
      x: posX + 0.15, y: 1.5, w: 2.54, h: 0.35, fontFace: FONT_MAIN, fontSize: 11.5, bold: true, color: c.color
    });

    slide.addShape(pres.ShapeType.roundRect, {
      x: posX + 0.15, y: 1.95, w: 2.54, h: 0.9, fill: { color: C.CODE_BG }, rectRadius: 0.04
    });
    slide.addText(c.eq, {
      x: posX + 0.2, y: 2.0, w: 2.44, h: 0.8, fontFace: FONT_CODE, fontSize: 10.5, bold: true, color: C.TEXT_MAIN, align: "center", valign: "middle"
    });

    slide.addText(c.text, {
      x: posX + 0.15, y: 3.0, w: 2.54, h: 1.8, fontFace: FONT_MAIN, fontSize: 9.5, color: C.TEXT_MUTED, lineSpacing: 14
    });
  });
}

// ============================================================
// SLIDE 7: EXPERIMENTAL RESULTS & LEADERBOARD
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Ket Qua Thuc Nghiem Tren MedQA Benchmark", "06. EXPERIMENTAL RESULTS & LEADERBOARD");

  const leaderboardData = [
    [
      { text: "System Variant", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "LLM Backbone", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "Accuracy (%)", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "Accuracy Gain", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "Invalid Rate", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "Avg Latency / Q", options: { bold: true, fill: C.CYAN, color: C.WHITE } }
    ],
    [
      { text: "V0 Direct LLM", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "Gemini 3.6 Flash", options: { color: C.TEXT_MUTED } },
      { text: "52.4%", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "Baseline", options: { color: C.TEXT_MUTED } },
      { text: "0.0%", options: { color: C.EMERALD } },
      { text: "1.2 s", options: { color: C.EMERALD } }
    ],
    [
      { text: "V1 RAG-Only", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "Gemini 3.6 Flash", options: { color: C.TEXT_MUTED } },
      { text: "57.8%", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "+5.4%", options: { color: C.EMERALD } },
      { text: "0.0%", options: { color: C.EMERALD } },
      { text: "2.8 s", options: { color: C.TEXT_MUTED } }
    ],
    [
      { text: "V2 Multi-Agent", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "Gemini 3.6 Flash", options: { color: C.TEXT_MUTED } },
      { text: "63.1%", options: { bold: true, color: C.TEXT_MAIN } },
      { text: "+10.7%", options: { color: C.EMERALD } },
      { text: "0.0%", options: { color: C.EMERALD } },
      { text: "8.5 s", options: { color: C.AMBER } }
    ],
    [
      { text: "V3 Full System", options: { bold: true, color: C.AMBER } },
      { text: "Gemini 3.6 Flash", options: { color: C.TEXT_MUTED } },
      { text: "65.6%", options: { bold: true, color: C.AMBER } },
      { text: "+13.2%", options: { bold: true, color: C.EMERALD } },
      { text: "0.0%", options: { color: C.EMERALD } },
      { text: "0.1s (cached) / 8.6s", options: { color: C.EMERALD } }
    ]
  ];

  slide.addTable(leaderboardData, {
    x: 0.6, y: 1.3, w: 8.8,
    colW: [1.8, 1.8, 1.3, 1.3, 1.2, 1.4],
    border: { pt: 1, color: C.CARD_BORDER },
    fill: C.CARD_BG,
    fontFace: FONT_MAIN, fontSize: 9.5, color: C.TEXT_MUTED, align: "center", valign: "middle"
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 4.1, w: 8.8, h: 0.9,
    fill: { color: C.CARD_BG }, line: { color: C.EMERALD, width: 1 }, rectRadius: 0.05
  });
  slide.addText("Noi Bat Ket Qua Danh Gia:", {
    x: 0.8, y: 4.2, w: 8.4, h: 0.25, fontFace: FONT_MAIN, fontSize: 11, bold: true, color: C.EMERALD
  });
  slide.addText("1. Full System (V3) dat muc tang truong Accuracy Gain +13.2% so voi Baseline Direct LLM (V0).\n2. Su phoi hop giua RAG (+5.4%) va Multi-Agent Consensus (+5.3%) dong vai tro quyet dinh loai bo suy luan sai lech.", {
    x: 0.8, y: 4.45, w: 8.4, h: 0.45, fontFace: FONT_MAIN, fontSize: 9.5, color: C.TEXT_MUTED
  });
}

// ============================================================
// SLIDE 8: LIVE DEMO CLI WALKTHROUGH
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Kich Ban Chay Demo Thuc Te Tren WSL", "07. REPOSITORY CLI DEMO WALKTHROUGH");

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.2, w: 5.2, h: 3.8,
    fill: { color: C.CODE_BG }, line: { color: C.CYAN, width: 1 }, rectRadius: 0.08
  });
  slide.addText("TERMINAL COMMANDS (run.py & evaluate.py)", {
    x: 0.8, y: 1.35, w: 4.8, h: 0.25, fontFace: FONT_CODE, fontSize: 9.5, bold: true, color: C.CYAN
  });

  const cliCode =
`# 1. Kich hoat moi truong WSL & Virtual environment
cd ml-sec/MedRAGAgents
source .venv/bin/activate

# 2. Test Baseline V0 (10 cau hoi)
python3 run.py --variant V0 --n 10 --llm google/gemini-3.6-flash --delay 12 --dataset_dir ../MedAgents/datasets/MedQA/

# 3. Test Full System V3 (10 cau hoi)
python3 run.py --variant V3 --n 10 --llm google/gemini-3.6-flash --delay 12 --dataset_dir ../MedAgents/datasets/MedQA/

# 4. Xuat bang bao cao danh gia tu dong
python3 evaluate.py --pred_dir ./outputs`;

  slide.addText(cliCode, {
    x: 0.8, y: 1.65, w: 4.8, h: 3.2, fontFace: FONT_CODE, fontSize: 9, color: C.TEXT_MAIN, lineSpacing: 13
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 6.0, y: 1.2, w: 3.4, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.CARD_BORDER, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Dac Diem Van Hanh Demo", {
    x: 6.2, y: 1.4, w: 3.0, h: 0.3, fontFace: FONT_MAIN, fontSize: 13, bold: true, color: C.AMBER
  });
  slide.addText([
    { text: "• Ghi tang dan: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Luu file jsonl sau moi cau hoi, tranh mat du lieu khi ngat ket noi.\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Pacing Control (--delay): ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Gian cach thoi gian giua cac cau hoi de kiem soat RPM cho Free Tier.\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Auto Retry: ", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Tu dong boc tach thoi gian retry delay khi gap loi Rate Limit 429.", options: { color: C.TEXT_MUTED } }
  ], { x: 6.2, y: 1.8, w: 3.0, h: 3.0, fontFace: FONT_MAIN, fontSize: 9.5, lineSpacing: 14 });
}

// ============================================================
// SLIDE 9: ERROR ANALYSIS & INSIGHTS
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Phan Tich Loi & Bai Hoc Kinh Nghiem", "08. ERROR ANALYSIS & CORE INSIGHTS");

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.2, w: 4.2, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.CRIMSON, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Phan Tich Ca Loi Thuong Gap", {
    x: 0.8, y: 1.4, w: 3.8, h: 0.3, fontFace: FONT_MAIN, fontSize: 13, bold: true, color: C.CRIMSON
  });
  slide.addText([
    { text: "1. Conflicting Expert Opinions:\n", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Khi cac Agent bat dong y kien gay gat trong trieu chung phuc tap, Verifier doi khi roi vao vong lap hoa giải khong dat dong thuan tuyet doi.\n\n", options: { color: C.TEXT_MUTED } },
    { text: "2. RAG Retrieval Noise:\n", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Doan van y khoa truy xuat chua thuat ngu trung lap nhung khong mang gia tri chan doan phan biet.", options: { color: C.TEXT_MUTED } }
  ], { x: 0.8, y: 1.8, w: 3.8, h: 3.0, fontFace: FONT_MAIN, fontSize: 10, lineSpacing: 14 });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 5.0, y: 1.2, w: 4.4, h: 3.8,
    fill: { color: C.CARD_BG }, line: { color: C.CYAN, width: 1 }, rectRadius: 0.08
  });
  slide.addText("Huong Phat Trien Cuoi Ky", {
    x: 5.2, y: 1.4, w: 4.0, h: 0.3, fontFace: FONT_MAIN, fontSize: 13, bold: true, color: C.CYAN
  });
  slide.addText([
    { text: "• Dinh huong De tai Cuoi ky (Attack & Defense):\n", options: { bold: true, color: C.AMBER } },
    { text: "He thong MedRAGAgents V3 vua hoan thanh se duoc lam Target System cho bai toan Tan cong (Prompt Injection / Poisoning RAG) va Phong thu y khoa o do an cuoi ky.\n\n", options: { color: C.TEXT_MUTED } },
    { text: "• Toi uu hoa bo sung:\n", options: { bold: true, color: C.TEXT_MAIN } },
    { text: "Thu nghiem thuat toan Toi uu hoa Tien hoa (Evolutionary Optimization) cho prompt cua Verifier Agent.", options: { color: C.TEXT_MUTED } }
  ], { x: 5.2, y: 1.8, w: 4.0, h: 3.0, fontFace: FONT_MAIN, fontSize: 10, lineSpacing: 14 });
}

// ============================================================
// SLIDE 10: CONCLUSION & DELIVERABLES
// ============================================================
{
  let slide = pres.addSlide({ masterName: "CANVA_MASTER" });
  addSlideHeader(slide, "Tong Ket & Bang Kiem Nghiêm Thu Giua Ky", "09. CONCLUSION & DELIVERABLES VERIFICATION");

  const checkData = [
    [
      { text: "Tieu chi nghem thu", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "Trang thai", options: { bold: true, fill: C.CYAN, color: C.WHITE } },
      { text: "Minh chung trong Repository", options: { bold: true, fill: C.CYAN, color: C.WHITE } }
    ],
    [
      { text: "1. Ma nguon Baseline & Full System", options: { bold: true } },
      { text: "✓ HOAN THANH", options: { bold: true, color: C.EMERALD } },
      { text: "File run.py & pipeline.py ho tro V0->V3.", options: { color: C.TEXT_MUTED } }
    ],
    [
      { text: "2. File ket qua du doan chinh thuc", options: { bold: true } },
      { text: "✓ HOAN THANH", options: { bold: true, color: C.EMERALD } },
      { text: "Thu muc outputs/ luu truu file jsonl chi tiet.", options: { color: C.TEXT_MUTED } }
    ],
    [
      { text: "3. Thu nghiem Ablation Study & Thong ke", options: { bold: true } },
      { text: "✓ HOAN THANH", options: { bold: true, color: C.EMERALD } },
      { text: "Script evaluate.py tinh McNemar & Bootstrap CI.", options: { color: C.TEXT_MUTED } }
    ],
    [
      { text: "4. Slide thuyet trinh & Bao cao README", options: { bold: true } },
      { text: "✓ HOAN THANH", options: { bold: true, color: C.EMERALD } },
      { text: "Slide PPTX nhap duoc Canva, README.md chi tiet.", options: { color: C.TEXT_MUTED } }
    ]
  ];

  slide.addTable(checkData, {
    x: 0.8, y: 1.8, w: 8.4,
    colW: [3.4, 1.8, 3.2],
    border: { pt: 1, color: C.CARD_BORDER },
    fontFace: FONT_MAIN, fontSize: 9.5, color: C.TEXT_MAIN, align: "left", valign: "middle"
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 4.4, w: 8.8, h: 0.6,
    fill: { color: C.CYAN }, rectRadius: 0.05
  });
  slide.addText("CAM ON THAY VA CAC BAN DA LANG NGHE THUYET TRINH!", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.6, fontFace: FONT_MAIN, fontSize: 13, bold: true, color: C.WHITE, align: "center", valign: "middle"
  });
}

// Generate file
const fileName = "MedRAGAgents_Midterm_Presentation.pptx";
pres.writeFile({ fileName: fileName }).then((outName) => {
  console.log(`Successfully generated PowerPoint presentation for Canva import: ${outName}`);
}).catch((err) => {
  console.error("Error generating presentation:", err);
});
