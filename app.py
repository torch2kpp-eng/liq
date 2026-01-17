import streamlit as st
import os
import io
import math
import random
import zipfile
from datetime import datetime
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="무지개 디저트 공방",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Fancy CSS (Kid-friendly)
# ---------------------------
APP_CSS = """
<style>
/* 전체 배경 */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 20% 10%, rgba(255,182,193,.35), transparent 45%),
              radial-gradient(circle at 90% 20%, rgba(135,206,235,.35), transparent 45%),
              radial-gradient(circle at 40% 90%, rgba(255,255,153,.30), transparent 45%),
              linear-gradient(180deg, #fff7fb 0%, #f6fbff 45%, #fffdf6 100%);
}

/* 사이드바 */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,.85), rgba(255,255,255,.75));
  border-right: 1px solid rgba(0,0,0,.06);
  backdrop-filter: blur(8px);
}

/* 타이틀 */
h1, h2, h3{
  letter-spacing: -0.5px;
}
.kid-card{
  background: rgba(255,255,255,.85);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,.06);
  padding: 18px 18px;
}
.kid-badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.9);
  border: 1px solid rgba(0,0,0,.06);
  font-size: 12px;
}
hr{
  border: none;
  height: 1px;
  background: rgba(0,0,0,.08);
  margin: 8px 0 16px 0;
}
.small-muted{
  color: rgba(0,0,0,.55);
  font-size: 13px;
}

/* 버튼 살짝 큼직하게 */
.stButton button, .stDownloadButton button{
  border-radius: 14px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(0,0,0,.10) !important;
  box-shadow: 0 8px 18px rgba(0,0,0,.06) !important;
}

/* expander */
.streamlit-expanderHeader{
  font-weight: 650;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def hex_to_rgb(hex_color: str):
    hex_color = hex_color.strip().lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def clamp(v, a=0, b=255):
    return max(a, min(b, v))

def load_font(font_size: int):
    # 선택: assets/fonts/*.ttf 있으면 사용, 없으면 기본 폰트
    candidates = [
        os.path.join("assets", "fonts", "NanumGothic.ttf"),
        os.path.join("assets", "fonts", "NanumGothicBold.ttf"),
        os.path.join("assets", "fonts", "Pretendard-Regular.ttf"),
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

@st.cache_data(show_spinner=False)
def load_base_image(image_path: str):
    img = Image.open(image_path).convert("RGBA")
    return img

def make_linear_gradient(size, c1, c2, direction="top-bottom"):
    w, h = size
    grad = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(grad)

    # direction에 따라 보간축 결정
    # t: 0~1
    if direction == "top-bottom":
        for y in range(h):
            t = y / max(1, (h - 1))
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b, 255))
    elif direction == "left-right":
        for x in range(w):
            t = x / max(1, (w - 1))
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.line([(x, 0), (x, h)], fill=(r, g, b, 255))
    elif direction == "diag":
        # 대각선 보간: (x+y)/(w+h)
        for y in range(h):
            for x in range(w):
                t = (x + y) / max(1, (w + h - 2))
                r = int(c1[0] + (c2[0] - c1[0]) * t)
                g = int(c1[1] + (c2[1] - c1[1]) * t)
                b = int(c1[2] + (c2[2] - c1[2]) * t)
                grad.putpixel((x, y), (r, g, b, 255))
    else:
        # default
        return make_linear_gradient(size, c1, c2, "top-bottom")

    return grad

def make_radial_gradient(size, c1, c2, center=(0.5, 0.4)):
    w, h = size
    cx, cy = center
    cx *= w
    cy *= h
    max_r = math.sqrt(max(cx, w-cx)**2 + max(cy, h-cy)**2)

    grad = Image.new("RGBA", (w, h))
    for y in range(h):
        for x in range(w):
            r = math.sqrt((x - cx)**2 + (y - cy)**2)
            t = min(1.0, r / max_r)
            rr = int(c1[0] + (c2[0] - c1[0]) * t)
            gg = int(c1[1] + (c2[1] - c1[1]) * t)
            bb = int(c1[2] + (c2[2] - c1[2]) * t)
            grad.putpixel((x, y), (rr, gg, bb, 255))
    return grad

def apply_gradient(img_rgba: Image.Image, color1_hex: str, color2_hex: str, mode="multiply", direction="top-bottom", radial=False):
    """디저트 질감을 살리면서 컬러를 입힘"""
    c1 = hex_to_rgb(color1_hex)
    c2 = hex_to_rgb(color2_hex)

    w, h = img_rgba.size
    if radial:
        grad = make_radial_gradient((w, h), c1, c2)
    else:
        grad = make_linear_gradient((w, h), c1, c2, direction=direction)

    if mode == "multiply":
        colored = ImageChops.multiply(img_rgba, grad)
    elif mode == "screen":
        colored = ImageChops.screen(img_rgba, grad)
    elif mode == "overlay":
        # 간단 overlay 근사: (multiply + screen)/2
        colored = Image.blend(ImageChops.multiply(img_rgba, grad), ImageChops.screen(img_rgba, grad), 0.5)
    else:
        colored = ImageChops.multiply(img_rgba, grad)

    out = Image.new("RGBA", img_rgba.size)
    out.paste(colored, (0, 0), mask=img_rgba.split()[-1])  # alpha 유지
    return out

def add_outline(img: Image.Image, outline_width=6, outline_color=(255, 255, 255, 255)):
    alpha = img.split()[-1]
    # 팽창(blur)로 외곽선 생성
    expanded = alpha.filter(ImageFilter.MaxFilter(size=max(3, outline_width*2+1)))
    outline = Image.new("RGBA", img.size, outline_color)
    outline.putalpha(expanded)
    # 원본 알파 영역 제외해서 테두리만
    outline_only = ImageChops.subtract(outline, Image.new("RGBA", img.size, (0,0,0,0)).putalpha(alpha) if False else outline)
    # 위 라인이 PIL에서 불편하므로 더 안전한 방식:
    # outline_only = outline - (outline masked by original alpha)
    mask_orig = Image.new("L", img.size, 0)
    mask_orig.paste(alpha, (0, 0))
    outline_mask = ImageChops.subtract(expanded, mask_orig)  # expanded - original
    outline_only = Image.new("RGBA", img.size, outline_color)
    outline_only.putalpha(outline_mask)
    return Image.alpha_composite(outline_only, img)

def add_drop_shadow(img: Image.Image, offset=(10, 14), blur=18, shadow_color=(0, 0, 0, 120)):
    w, h = img.size
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    alpha = img.split()[-1]
    shadow_layer = Image.new("RGBA", (w, h), shadow_color)
    shadow_layer.putalpha(alpha)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur))

    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.alpha_composite(shadow_layer, dest=offset)
    base.alpha_composite(img, dest=(0, 0))
    return base

def sprinkle_layer(size, density=120, seed=42, palette=None, min_len=6, max_len=18, width=3, alpha=210):
    """스프링클/별가루 레이어 생성"""
    w, h = size
    rnd = random.Random(seed)

    if palette is None:
        palette = [
            (255, 182, 193), (135, 206, 235), (255, 255, 153),
            (186, 255, 201), (221, 160, 221), (255, 215, 0)
        ]

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for _ in range(density):
        x = rnd.randint(0, w-1)
        y = rnd.randint(0, h-1)
        c = palette[rnd.randint(0, len(palette)-1)]
        c = (c[0], c[1], c[2], alpha)

        kind = rnd.choice(["line", "dot", "star"])
        if kind == "dot":
            r = rnd.randint(2, 5)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=c)
        elif kind == "line":
            L = rnd.randint(min_len, max_len)
            ang = rnd.random() * math.pi
            x2 = int(x + L * math.cos(ang))
            y2 = int(y + L * math.sin(ang))
            draw.line((x, y, x2, y2), fill=c, width=width)
        else:  # star
            r = rnd.randint(6, 12)
            pts = []
            for k in range(10):
                rr = r if k % 2 == 0 else r * 0.45
                a = (math.pi/5) * k
                pts.append((x + rr*math.cos(a), y + rr*math.sin(a)))
            draw.polygon(pts, fill=c)

    return layer

def make_sticker(kind="star", size=(220, 220), fill=(255, 255, 255, 230), stroke=(0,0,0,80), stroke_w=6):
    """기본 도형 스티커 생성"""
    w, h = size
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)

    if kind == "star":
        cx, cy = w//2, h//2
        R = min(w, h)*0.40
        r = R*0.45
        pts = []
        for k in range(10):
            rr = R if k % 2 == 0 else r
            a = -math.pi/2 + k*(math.pi/5)
            pts.append((cx + rr*math.cos(a), cy + rr*math.sin(a)))
        d.polygon(pts, fill=fill, outline=stroke)
    elif kind == "heart":
        # 간단 하트
        cx, cy = w//2, h//2
        s = min(w,h)*0.34
        pts = []
        for t in [i/200 for i in range(0, 201)]:
            x = 16*math.sin(t*math.pi*2)**3
            y = 13*math.cos(t*math.pi*2) - 5*math.cos(2*t*math.pi*2) - 2*math.cos(3*t*math.pi*2) - math.cos(4*t*math.pi*2)
            pts.append((cx + x*s/18, cy - y*s/18))
        d.polygon(pts, fill=fill, outline=stroke)
    elif kind == "rainbow":
        # 무지개 아치
        pad = int(min(w,h)*0.10)
        bbox = (pad, pad, w-pad, h-pad)
        bands = [
            (255, 99, 132, 220),
            (255, 159, 64, 220),
            (255, 205, 86, 220),
            (75, 192, 192, 220),
            (54, 162, 235, 220),
            (153, 102, 255, 220),
        ]
        thick = int(min(w,h)*0.10)
        for i, col in enumerate(bands):
            bb = (bbox[0]+i*thick//3, bbox[1]+i*thick//3, bbox[2]-i*thick//3, bbox[3]-i*thick//3)
            d.arc(bb, start=200, end=340, fill=col, width=thick)
        # 구름
        cloud = (255,255,255,240)
        for dx in [-40, 40]:
            x0 = w//2 + dx - 60
            y0 = h//2 + 40
            d.ellipse((x0, y0, x0+110, y0+70), fill=cloud)
            d.ellipse((x0+25, y0-20, x0+95, y0+55), fill=cloud)
    else:
        d.rounded_rectangle((20, 20, w-20, h-20), radius=28, fill=fill, outline=stroke, width=stroke_w)

    # 외곽선 두껍게 느낌
    if stroke_w > 0:
        img = img.filter(ImageFilter.GaussianBlur(0.2))
    return img

def paste_centered(base: Image.Image, overlay: Image.Image, center_xy, scale=1.0, rotation=0):
    ov = overlay.copy()
    if scale != 1.0:
        nw = max(1, int(ov.size[0] * scale))
        nh = max(1, int(ov.size[1] * scale))
        ov = ov.resize((nw, nh), resample=Image.Resampling.LANCZOS)

    if rotation != 0:
        ov = ov.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    x = int(center_xy[0] - ov.size[0] / 2)
    y = int(center_xy[1] - ov.size[1] / 2)
    base.alpha_composite(ov, dest=(x, y))
    return base

def add_text_label(img: Image.Image, text: str, pos=("center", "bottom"), font_size=48,
                   fill=(30,30,30,255), stroke_fill=(255,255,255,230), stroke_w=4,
                   pad=16, bubble=True):
    if not text.strip():
        return img

    w, h = img.size
    font = load_font(font_size)
    draw = ImageDraw.Draw(img)

    # 텍스트 박스 크기
    bbox = draw.textbbox((0,0), text, font=font, stroke_width=stroke_w)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # 위치 계산
    if pos[0] == "left":
        x = int(0.08*w)
    elif pos[0] == "right":
        x = int(0.92*w - tw)
    else:
        x = int((w - tw) / 2)

    if pos[1] == "top":
        y = int(0.06*h)
    elif pos[1] == "center":
        y = int((h - th) / 2)
    else:
        y = int(0.90*h - th)

    # 말풍선/라벨 배경
    if bubble:
        bx0 = x - pad
        by0 = y - pad
        bx1 = x + tw + pad
        by1 = y + th + pad
        bubble_bg = Image.new("RGBA", img.size, (0,0,0,0))
        bd = ImageDraw.Draw(bubble_bg)
        bd.rounded_rectangle((bx0, by0, bx1, by1), radius=18, fill=(255,255,255,210), outline=(0,0,0,50), width=2)
        bubble_bg = bubble_bg.filter(ImageFilter.GaussianBlur(0.4))
        img = Image.alpha_composite(img, bubble_bg)
        draw = ImageDraw.Draw(img)

    draw.text((x, y), text, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_w)
    return img

def export_png_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def safe_filename(s: str):
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))[:80]

# ---------------------------
# Assets
# ---------------------------
dessert_options = {
    "아이스크림": "icecream.png",
    "빙수": "shaved_ice.png",
    "젤리빈": "jellybean.png",
    "마카롱": "macaron.png",
    "쿠키": "cookie.png",
}

# ---------------------------
# Session State
# ---------------------------
if "gallery" not in st.session_state:
    st.session_state.gallery = []  # list of dict: {name, meta, bytes, ts}

if "seed" not in st.session_state:
    st.session_state.seed = 42

# ---------------------------
# Header
# ---------------------------
left, right = st.columns([3, 2], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div class="kid-card">
          <div class="kid-badge">🍭 Rainbow Dessert Studio</div>
          <h1 style="margin: 10px 0 4px 0;">무지개 디저트 공방</h1>
          <div class="small-muted">색을 고르고, 토핑을 뿌리고, 스티커와 글자를 붙여서 나만의 디저트를 완성해보세요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="kid-card">
          <div class="kid-badge">팁</div>
          <div class="small-muted" style="margin-top:8px;">
            1) 그라데이션 모드는 Multiply가 질감이 가장 자연스럽습니다.<br/>
            2) 스프링클 밀도를 올리면 더 화려해집니다.<br/>
            3) 작품은 아래 갤러리에 저장할 수 있습니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("🎨 디자인 센터")

# Step selector
step = st.sidebar.radio(
    "꾸미기 단계",
    ["1) 디저트 선택", "2) 색(그라데이션)", "3) 토핑(스프링클)", "4) 스티커 & 글자", "5) 저장 & 내보내기"],
)

st.sidebar.divider()

# Common: Dessert selection
selected_name = st.sidebar.selectbox("디저트를 골라보세요", list(dessert_options.keys()))
image_filename = dessert_options[selected_name]
image_path = os.path.join("assets", image_filename)

# Random recipe button
if st.sidebar.button("🎲 랜덤 레시피 뽑기"):
    st.session_state.seed = random.randint(1, 999999)
    st.sidebar.success("랜덤 레시피를 적용했습니다. 아래 옵션을 확인해보세요.")

# Color presets
preset_palettes = {
    "핑크-하늘": ("#FFB6C1", "#87CEEB"),
    "레몬-민트": ("#FFF59D", "#A7FFEB"),
    "라벤더-피치": ("#C7B8FF", "#FFD1A6"),
    "딸기-바나나": ("#FF8FA3", "#FFE082"),
    "초코-크림": ("#A1887F", "#FFF3E0"),
}

st.sidebar.subheader("🌈 색상")
preset = st.sidebar.selectbox("팔레트 프리셋", ["직접 선택"] + list(preset_palettes.keys()))
if preset != "직접 선택":
    default_top, default_bottom = preset_palettes[preset]
else:
    default_top, default_bottom = "#FFB6C1", "#87CEEB"

color_top = st.sidebar.color_picker("윗부분 색상", default_top)
color_bottom = st.sidebar.color_picker("아랫부분 색상", default_bottom)

with st.sidebar.expander("고급 색 옵션", expanded=False):
    gradient_mode = st.selectbox("합성 모드", ["multiply", "overlay", "screen"], index=0)
    gradient_direction = st.selectbox("방향", ["top-bottom", "left-right", "diag"], index=0)
    radial = st.checkbox("라디얼(원형) 그라데이션", value=False)

st.sidebar.subheader("✨ 꾸미기")
with st.sidebar.expander("외곽선/그림자/프레임", expanded=False):
    use_shadow = st.checkbox("그림자", value=True)
    shadow_blur = st.slider("그림자 블러", 0, 40, 18, 1)
    shadow_dx = st.slider("그림자 X", -30, 30, 10, 1)
    shadow_dy = st.slider("그림자 Y", -30, 30, 14, 1)

    use_outline = st.checkbox("하이라이트 외곽선", value=True)
    outline_w = st.slider("외곽선 두께", 0, 20, 6, 1)
    outline_col = st.color_picker("외곽선 색", "#FFFFFF")

    use_frame = st.checkbox("액자 프레임", value=False)
    frame_thick = st.slider("프레임 두께", 10, 120, 50, 2)
    frame_col = st.color_picker("프레임 색", "#FFFFFF")

with st.sidebar.expander("토핑(스프링클)", expanded=False):
    sprinkles_on = st.checkbox("스프링클 뿌리기", value=True)
    sprinkles_density = st.slider("밀도", 0, 600, 180, 10)
    sprinkles_alpha = st.slider("투명도", 50, 255, 210, 5)
    sprinkles_size = st.slider("크기감(선 두께)", 1, 8, 3, 1)
    sprinkles_seed = st.number_input("랜덤 시드", min_value=1, max_value=999999, value=int(st.session_state.seed), step=1)

with st.sidebar.expander("스티커 & 글자", expanded=False):
    sticker_kind = st.selectbox("기본 스티커", ["없음", "star", "heart", "rainbow"], index=1)
    sticker_scale = st.slider("스티커 크기", 0.3, 2.2, 1.0, 0.05)
    sticker_rot = st.slider("스티커 회전", -180, 180, 0, 5)
    sticker_x = st.slider("스티커 위치 X", 0, 100, 75, 1)  # %
    sticker_y = st.slider("스티커 위치 Y", 0, 100, 20, 1)  # %
    uploaded_sticker = st.file_uploader("내 스티커 PNG 업로드(투명 배경)", type=["png"])

    label_text = st.text_input("작품 이름/메시지", value=f"내 {selected_name}!")
    label_pos_x = st.selectbox("글자 위치(좌우)", ["center", "left", "right"], index=0)
    label_pos_y = st.selectbox("글자 위치(상하)", ["bottom", "top", "center"], index=0)
    label_font = st.slider("글자 크기", 16, 96, 44, 2)
    label_color = st.color_picker("글자 색", "#1E1E1E")
    label_outline = st.checkbox("글자 테두리", value=True)
    label_bubble = st.checkbox("말풍선 배경", value=True)

st.sidebar.divider()

with st.sidebar.expander("내보내기", expanded=False):
    export_scale = st.slider("내보내기 해상도 배율", 1, 3, 2, 1)
    include_recipe = st.checkbox("레시피 카드 포함(ZIP)", value=True)

# ---------------------------
# Build Image Pipeline
# ---------------------------
def render_dessert():
    if not os.path.exists(image_path):
        # 자산이 없을 때: 임시 플레이스홀더 생성
        ph = Image.new("RGBA", (900, 700), (255, 255, 255, 0))
        d = ImageDraw.Draw(ph)
        d.rounded_rectangle((120, 120, 780, 580), radius=60, fill=(255,255,255,220), outline=(0,0,0,30), width=3)
        d.text((160, 320), "assets 폴더에\n디저트 PNG가 필요해요.", fill=(0,0,0,180), font=load_font(36))
        base = ph
    else:
        base = load_base_image(image_path)

    # 1) Gradient
    out = apply_gradient(
        base,
        color_top,
        color_bottom,
        mode=gradient_mode,
        direction=gradient_direction,
        radial=radial
    )

    # 2) Outline
    if use_outline and outline_w > 0:
        oc = hex_to_rgb(outline_col) + (255,)
        out = add_outline(out, outline_width=outline_w, outline_color=oc)

    # 3) Sprinkles (only on opaque regions)
    if sprinkles_on and sprinkles_density > 0:
        pal = [hex_to_rgb(color_top), hex_to_rgb(color_bottom), (255,255,153), (186,255,201), (221,160,221), (255,215,0)]
        sp = sprinkle_layer(out.size, density=sprinkles_density, seed=int(sprinkles_seed),
                            palette=pal, width=int(sprinkles_size), alpha=int(sprinkles_alpha))
        # 알파 영역에만 스프링클 보이도록 마스킹
        alpha = out.split()[-1]
        sp.putalpha(ImageChops.multiply(sp.split()[-1], alpha))
        out = Image.alpha_composite(out, sp)

    # 4) Sticker
    if uploaded_sticker is not None:
        try:
            st_img = Image.open(uploaded_sticker).convert("RGBA")
        except Exception:
            st_img = None
    else:
        st_img = None

    if sticker_kind != "없음" or st_img is not None:
        if st_img is None:
            st_img = make_sticker(kind=sticker_kind, size=(240, 240), fill=(255,255,255,235), stroke=(0,0,0,70), stroke_w=6)

        cx = int(out.size[0] * (sticker_x / 100))
        cy = int(out.size[1] * (sticker_y / 100))
        out = paste_centered(out, st_img, (cx, cy), scale=float(sticker_scale), rotation=int(sticker_rot))

    # 5) Text label
    fill = hex_to_rgb(label_color) + (255,)
    stroke_fill = (255,255,255,230) if label_outline else None
    out = add_text_label(
        out,
        text=label_text,
        pos=(label_pos_x, label_pos_y),
        font_size=int(label_font),
        fill=fill,
        stroke_fill=stroke_fill if stroke_fill else (0,0,0,0),
        stroke_w=4 if label_outline else 0,
        bubble=bool(label_bubble),
    )

    # 6) Shadow (마지막에)
    if use_shadow:
        out = add_drop_shadow(out, offset=(int(shadow_dx), int(shadow_dy)), blur=int(shadow_blur), shadow_color=(0,0,0,120))

    # 7) Frame (마지막 장식)
    if use_frame:
        w, h = out.size
        fr = Image.new("RGBA", (w + frame_thick*2, h + frame_thick*2), (0,0,0,0))
        d = ImageDraw.Draw(fr)
        fc = hex_to_rgb(frame_col) + (255,)
        d.rounded_rectangle((0,0, fr.size[0]-1, fr.size[1]-1), radius=28, fill=fc, outline=(0,0,0,30), width=3)
        fr.alpha_composite(out, dest=(frame_thick, frame_thick))
        out = fr

    return out

result_img = render_dessert()

# ---------------------------
# Main Layout
# ---------------------------
colA, colB = st.columns([3.2, 1.8], gap="large")

with colA:
    st.markdown('<div class="kid-card">', unsafe_allow_html=True)
    st.subheader("🍰 미리보기")
    st.image(result_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="kid-card">', unsafe_allow_html=True)
    st.subheader("🧾 나의 레시피 카드")

    st.write(f"- **디저트:** {selected_name}")
    st.write(f"- **그라데이션:** {color_top} → {color_bottom} ({'라디얼' if radial else gradient_direction}, {gradient_mode})")
    st.write(f"- **스프링클:** {'ON' if sprinkles_on else 'OFF'} / 밀도 {sprinkles_density} / 시드 {sprinkles_seed}")
    st.write(f"- **스티커:** {'업로드' if uploaded_sticker else sticker_kind}")
    st.write(f"- **텍스트:** {label_text}")

    st.write("적용된 그라데이션 미리보기:")
    st.markdown(
        f"""
        <div style="
            width: 100%;
            height: 68px;
            border-radius: 16px;
            background: linear-gradient(to bottom, {color_top}, {color_bottom});
            border: 1px solid rgba(0,0,0,.08);
            box-shadow: 0 8px 18px rgba(0,0,0,.06);
        "></div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # 다운로드(단일)
    export_img = result_img
    if export_scale != 1:
        export_img = export_img.resize(
            (export_img.size[0]*export_scale, export_img.size[1]*export_scale),
            resample=Image.Resampling.LANCZOS
        )

    out_bytes = export_png_bytes(export_img)
    st.download_button(
        label="🖼️ PNG 저장하기",
        data=out_bytes,
        file_name=f"my_{safe_filename(selected_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        use_container_width=True
    )

    # 저장(갤러리)
    if st.button("⭐ 내 갤러리에 저장", use_container_width=True):
        st.session_state.gallery.append({
            "name": label_text.strip() if label_text.strip() else f"내 {selected_name}",
            "dessert": selected_name,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "meta": {
                "top": color_top,
                "bottom": color_bottom,
                "mode": gradient_mode,
                "direction": gradient_direction,
                "radial": radial,
                "sprinkles": sprinkles_on,
                "sprinkles_density": sprinkles_density,
                "sprinkles_seed": int(sprinkles_seed),
                "sticker": "uploaded" if uploaded_sticker else sticker_kind,
            },
            "png": export_png_bytes(result_img)  # 원본 크기 저장
        })
        st.balloons()
        st.success("갤러리에 저장했습니다.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.divider()

# ---------------------------
# Gallery Section
# ---------------------------
st.markdown('<div class="kid-card">', unsafe_allow_html=True)
st.subheader("🖼️ 내 작품 갤러리")

if len(st.session_state.gallery) == 0:
    st.info("아직 저장된 작품이 없습니다. 위에서 ‘내 갤러리에 저장’을 눌러보세요.")
else:
    # ZIP 다운로드 준비
    if include_recipe:
        # 레시피 텍스트도 함께
        pass

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, item in enumerate(st.session_state.gallery, start=1):
            base_name = safe_filename(item["name"]) or f"dessert_{i}"
            zf.writestr(f"{i:02d}_{base_name}.png", item["png"])

            if include_recipe:
                meta = item["meta"]
                recipe_txt = (
                    f"작품명: {item['name']}\n"
                    f"디저트: {item['dessert']}\n"
                    f"저장시간: {item['ts']}\n"
                    f"그라데이션: {meta['top']} -> {meta['bottom']}\n"
                    f"모드: {meta['mode']}\n"
                    f"방향: {meta['direction']}\n"
                    f"라디얼: {meta['radial']}\n"
                    f"스프링클: {meta['sprinkles']} / 밀도 {meta['sprinkles_density']} / 시드 {meta['sprinkles_seed']}\n"
                    f"스티커: {meta['sticker']}\n"
                )
                zf.writestr(f"{i:02d}_{base_name}_recipe.txt", recipe_txt)

    st.download_button(
        label="📦 갤러리 ZIP 다운로드",
        data=zip_buf.getvalue(),
        file_name=f"dessert_gallery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=False
    )

    # 썸네일 표시
    cols = st.columns(4)
    for idx, item in enumerate(reversed(st.session_state.gallery)):
        c = cols[idx % 4]
        with c:
            img = Image.open(io.BytesIO(item["png"])).convert("RGBA")
            st.image(img, use_container_width=True)
            st.caption(f"{item['name']}  ({item['ts']})")
            st.download_button(
                label="PNG 받기",
                data=item["png"],
                file_name=f"{safe_filename(item['name'])}.png",
                mime="image/png",
                key=f"dl_{idx}"
            )

    st.write("")
    if st.button("🧹 갤러리 비우기"):
        st.session_state.gallery = []
        st.success("갤러리를 비웠습니다.")

st.markdown("</div>", unsafe_allow_html=True)
