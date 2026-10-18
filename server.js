import express from "express";
import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { InferenceClient } from "@huggingface/inference";

dotenv.config();

const { HF_TOKEN, CF_ACCOUNT_ID, CF_API_TOKEN } = process.env;

// Провайдер: "cloudflare" (безкоштовний щоденний ліміт) або "huggingface"
const PROVIDER = (process.env.PROVIDER || (CF_API_TOKEN ? "cloudflare" : "huggingface")).toLowerCase();

const need = PROVIDER === "cloudflare" ? ["CF_ACCOUNT_ID", "CF_API_TOKEN"] : ["HF_TOKEN"];
const missing = need.filter(k => !process.env[k]);
if (missing.length) {
    console.error(`❌ У файлі .env немає: ${missing.join(", ")}. Дивись .env.example`);
    process.exit(1);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;

const HF_IMAGE_MODEL = process.env.IMAGE_MODEL || "black-forest-labs/FLUX.1-schnell";
const HF_CHAT_MODEL = process.env.CHAT_MODEL || "Qwen/Qwen2.5-72B-Instruct";
const CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell";
const CF_CHAT_MODEL = process.env.CF_CHAT_MODEL || "@cf/meta/llama-3.1-8b-instruct";

const hf = HF_TOKEN ? new InferenceClient(HF_TOKEN) : null;
console.log(`Провайдер: ${PROVIDER}`);

const app = express();
// За хостингом (Render, Railway тощо) справжню IP-адресу відвідувача дає проксі
app.set("trust proxy", 1);
app.use(express.json({ limit: "10kb" }));
app.use(express.static(path.join(__dirname, "public")));

// ---------- допоміжне ----------

const hits = new Map();
setInterval(() => hits.clear(), 10 * 60 * 1000);
const limiter = (max, windowMs) => (req, res, next) => {
    const now = Date.now();
    const list = (hits.get(req.ip) || []).filter(t => now - t < windowMs);
    if (list.length >= max) {
        return res.status(429).json({ error: "Забагато запитів. Зачекай хвилинку." });
    }
    list.push(now);
    hits.set(req.ip, list);
    next();
};

const clean = (s, max) => String(s ?? "").replace(/\s+/g, " ").trim().slice(0, max);
const cleanWords = w => Array.isArray(w) ? w.slice(0, 3).map(x => clean(x, 30)).filter(Boolean) : [];

function parseJson(text) {
    const m = String(text).match(/\{[\s\S]*\}/);
    if (!m) throw new Error("AI повернув не JSON");
    return JSON.parse(m[0].replace(/\s*\n\s*/g, " "));
}

function fail(res, error, fallback) {
    console.error(error);
    const m = String(error?.message || "");
    let msg = fallback;
    if (/401|403|unauthorized|token|authentication/i.test(m)) {
        msg = "Ключ доступу не підійшов. Перевір дані в файлі .env.";
    } else if (/402|429|quota|credit|rate|limit|allocation|neurons/i.test(m)) {
        msg = "Безкоштовний ліміт вичерпано. Спробуй пізніше (у Cloudflare він оновлюється щодня).";
    }
    res.status(500).json({ error: msg });
}

// ---------- провайдери ----------

async function cf(model, body) {
    const r = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/ai/run/${model}`,
        {
            method: "POST",
            headers: { Authorization: `Bearer ${CF_API_TOKEN}`, "Content-Type": "application/json" },
            body: JSON.stringify(body)
        }
    );
    const d = await r.json().catch(() => ({}));
    if (!r.ok || d.success === false) {
        throw new Error(`${r.status} ${d.errors?.[0]?.message || "помилка Cloudflare"}`);
    }
    return d.result;
}

async function ask(system, user, maxTokens = 400) {
    const messages = [
        { role: "system", content: system },
        { role: "user", content: user }
    ];
    if (PROVIDER === "cloudflare") {
        const r = await cf(CF_CHAT_MODEL, { messages, max_tokens: maxTokens });
        // Cloudflare інколи віддає відповідь уже як об'єкт, якщо вона схожа на JSON
        const out = r.response ?? r.choices?.[0]?.message?.content ?? "";
        return (typeof out === "string" ? out : JSON.stringify(out)).trim();
    }
    const r = await hf.chatCompletion({ model: HF_CHAT_MODEL, messages, max_tokens: maxTokens, temperature: 0.7 });
    return r.choices?.[0]?.message?.content?.trim() || "";
}

async function draw(prompt) {
    if (PROVIDER === "cloudflare") {
        const r = await cf(CF_IMAGE_MODEL, { prompt: prompt.slice(0, 2000), steps: 4 });
        return `data:image/jpeg;base64,${r.image}`;
    }
    const image = await hf.textToImage({ model: HF_IMAGE_MODEL, inputs: prompt });
    const buffer = Buffer.from(await image.arrayBuffer());
    return `data:${image.type || "image/png"};base64,${buffer.toString("base64")}`;
}

// Просить AI відповісти JSON; якщо вийшла дурниця, пробує ще раз
async function askJson(system, user, maxTokens = 400) {
    let lastError;
    for (let i = 0; i < 2; i++) {
        try {
            return parseJson(await ask(system, user, maxTokens));
        } catch (e) {
            lastError = e;
            console.warn(`Спроба ${i + 1}: AI відповів некоректно (${e.message})`);
        }
    }
    throw lastError;
}

// ---------- стилі ----------

// Ключі мають збігатися зі стилями у public/index.html
const STYLES = {
    auto: "High quality digital artwork, interesting composition and lighting.",
    cinematic: "Cinematic film still, dramatic lighting, shallow depth of field.",
    watercolor: "Soft watercolor painting, visible paper texture, gentle colors.",
    anime: "Anime illustration, clean line art, vivid colors.",
    pixel: "Detailed pixel art, 16-bit game style.",
    surreal: "Surrealist painting in the spirit of Dalí and Magritte, dreamlike.",
    clay: "Cute 3D clay render, soft studio light, tiny details."
};
// Шаблон навмисно малюємо нудним стоковим стилем, щоб було видно його «звичайність»
const STOCK = "Generic corporate stock illustration, flat colors, plain centered composition, bland and predictable.";
const NO_TEXT = "No text, letters or captions in the image.";

// ---------- API ----------

const SYS_WORDS = `You are a creativity tool. The user gives three random words. Imagine the MOST OBVIOUS, predictable idea that 90% of people would invent by combining them: the first thing that comes to mind.
Answer ONLY with JSON: {"ua":"1-2 sentences in Ukrainian describing this obvious idea","en":"a vivid English image prompt of this scene, max 60 words"}`;

const SYS_PROBLEM = `You are a creativity tool. The user describes a work problem. Imagine the MOST OBVIOUS, predictable solution that almost every team would propose first.
Answer ONLY with JSON: {"ua":"1-2 sentences in Ukrainian describing this obvious solution","en":"an English image prompt that shows this solution as a simple scene or metaphor, max 60 words"}`;

// Крок 1: «шаблон», тобто найочевидніша ідея + її картинка
app.post("/api/cliche", limiter(8, 60_000), async (req, res) => {
    const mode = req.body?.mode === "problem" ? "problem" : "words";
    const words = cleanWords(req.body?.words);
    const problem = clean(req.body?.problem, 400);

    if (mode === "words" && words.length < 3) {
        return res.status(400).json({ error: "Потрібні три слова." });
    }
    if (mode === "problem" && problem.length < 10) {
        return res.status(400).json({ error: "Опиши проблему хоча б одним реченням." });
    }

    try {
        const d = await askJson(
            mode === "words" ? SYS_WORDS : SYS_PROBLEM,
            mode === "words" ? `Words: ${words.join(", ")}` : `Problem: ${problem}`,
            300
        );
        const ua = clean(d.ua, 300), en = clean(d.en, 500);
        if (!ua || !en) throw new Error("AI повернув неповну відповідь");

        const image = await draw(`${en} ${STOCK} ${NO_TEXT}`);
        res.json({ image, ua });
    } catch (error) {
        fail(res, error, "Не вдалося показати шаблон. Спробуй ще раз.");
    }
});

// Крок 2: картинка для ідеї користувача
async function buildPrompt(idea, words, style) {
    let scene = idea;
    try {
        if (process.env.ENHANCE_PROMPT === "false") throw new Error("вимкнено в .env");
        const t = await ask(
            "You write prompts for an image generator. Translate the user's idea to English and turn it into one vivid scene description (max 70 words): subject, setting, lighting, composition. Keep every concrete detail the user mentioned. Output only the prompt, without quotes.",
            `Idea: ${idea}${words.length ? `\nInspiration words: ${words.join(", ")}` : ""}`,
            200
        );
        if (t) scene = t;
    } catch (e) {
        console.warn("Покращення промпту пропущено:", e.message);
    }
    return `${scene} ${STYLES[style]} ${NO_TEXT}`;
}

app.post("/api/generate", limiter(8, 60_000), async (req, res) => {
    const idea = clean(req.body?.idea, 600);
    const words = cleanWords(req.body?.words);

    if (idea.length < 3) {
        return res.status(400).json({ error: "Напиши свою ідею (мінімум 3 символи)." });
    }
    const style = STYLES[req.body?.style] ? req.body.style : "auto";

    try {
        const prompt = await buildPrompt(idea, words, style);
        const image = await draw(prompt);
        res.json({ success: true, image, prompt });
    } catch (error) {
        fail(res, error, "Не вдалося створити зображення.");
    }
});

// Крок 3: наскільки ідея далека від шаблону
app.post("/api/compare", limiter(10, 60_000), async (req, res) => {
    const topic = clean(req.body?.topic, 400);
    const cliche = clean(req.body?.cliche, 400);
    const idea = clean(req.body?.idea, 600);

    if (!topic || !cliche || idea.length < 3) {
        return res.status(400).json({ error: "Не вистачає даних для порівняння." });
    }

    try {
        const d = await askJson(
            `You are a strict but friendly creativity coach. Reply in Ukrainian.
You get a topic, the OBVIOUS idea that everyone would propose, and the user's idea. Judge how far the user's idea is from the obvious one.
distance: 1-3 = the same idea with decoration; 4-6 = same core with a twist; 7-10 = a fundamentally different concept.
Answer ONLY with JSON:
{"distance":1-10,"verdict":"1-2 sentences, honest, no flattery","still_obvious":"what in the idea still repeats the obvious one (or empty if nothing)","push":"one concrete direction to go further"}`,
            `Topic: ${topic}\nObvious idea: ${cliche}\nUser's idea: ${idea}`,
            400
        );
        res.json({
            distance: Math.min(10, Math.max(1, Math.round(Number(d.distance)) || 5)),
            verdict: clean(d.verdict, 400),
            still: clean(d.still_obvious, 300),
            push: clean(d.push, 300)
        });
    } catch (error) {
        fail(res, error, "Не вдалося порівняти ідеї. Спробуй ще раз.");
    }
});

app.use((err, req, res, next) => {
    res.status(400).json({ error: "Некоректний запит." });
});

app.listen(PORT, () => {
    console.log(`Creative Lab працює на http://localhost:${PORT}`);
});