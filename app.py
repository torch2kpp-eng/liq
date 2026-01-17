import streamlit as st
import os
from PIL import Image, ImageChops, ImageDraw

# 페이지 설정
st.set_page_config(page_title="무지개 디저트 공방", page_icon="🌈")

def apply_gradient_to_image(image_path, color1_hex, color2_hex):
    """이미지에 상단->하단 그라데이션을 입히는 함수"""
    try:
        # 이미지 불러오기 및 RGBA 변환
        img = Image.open(image_path).convert("RGBA")
        width, height = img.size
        
        # Hex 색상을 RGB로 변환
        c1 = tuple(int(color1_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        c2 = tuple(int(color2_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # 1. 그라데이션 레이어 생성
        gradient = Image.new('RGBA', (width, height))
        draw = ImageDraw.Draw(gradient)
        
        for y in range(height):
            # 상단(c1)에서 하단(c2)으로 색상 보간 계산
            r = int(c1[0] + (c2[0] - c1[0]) * (y / height))
            g = int(c1[1] + (c2[1] - c1[1]) * (y / height))
            b = int(c1[2] + (c2[2] - c1[2]) * (y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
        
        # 2. Multiply(승산) 합성을 통해 디저트 질감 위에 색상 적용
        colored_part = ImageChops.multiply(img, gradient)
        
        # 3. 원본 이미지의 알파 채널(투명도)을 유지하며 합성
        output = Image.new("RGBA", img.size)
        output.paste(colored_part, (0, 0), mask=img)
        return output
    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")
        return None

# --- UI 구성 ---
st.title("🌈 나만의 그라데이션 디저트")
st.write("두 가지 색깔을 골라 세상에 하나뿐인 무지개 디저트를 만들어봐요!")

# 사이드바: 설정 도구
st.sidebar.header("🎨 디자인 센터")

dessert_options = {
    "아이스크림": "icecream.png",
    "빙수": "shaved_ice.png",
    "젤리빈": "jellybean.png",
    "마카롱": "macaron.png",
    "쿠키": "cookie.png"
}

selected_name = st.sidebar.selectbox("디저트를 골라보세요", list(dessert_options.keys()))
color_top = st.sidebar.color_picker("윗부분 색상", "#FFB6C1")   # 기본 핑크
color_bottom = st.sidebar.color_picker("아랫부분 색상", "#87CEEB") # 기본 하늘색

st.divider()

# --- 메인 실행 로직 ---
image_filename = dessert_options[selected_name]
image_path = os.path.join("assets", image_filename)

if os.path.exists(image_path):
    # 그라데이션 합성 함수 호출
    result_img = apply_gradient_to_image(image_path, color_top, color_bottom)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if result_img:
            st.image(result_img, caption=f"멋진 {selected_name} 완성!", use_container_width=True)
    
    with col2:
        st.subheader("📝 나의 레시피")
        st.info(f"선택한 디저트: **{selected_name}**")
        
        # 선택한 그라데이션 미리보기 박스 (CSS 사용)
        st.write("적용된 그라데이션:")
        st.markdown(f"""
            <div style="
                width: 100%; 
                height: 60px; 
                border-radius: 15px; 
                background: linear-gradient(to bottom, {color_top}, {color_bottom});
                border: 2px solid #f0f0f0;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            "></div>
        """, unsafe_allow_html=True)
        
        # 이미지 다운로드 기능
        import io
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="🖼️ 이미지 저장하기",
            data=buf.getvalue(),
            file_name=f"my_{selected_name}.png",
            mime="image/png"
        )

else:
    st.warning(f"assets 폴더에 '{image_filename}' 파일이 있는지 확인해주세요!")

# 완료 효과
if st.button("🎉 작품 완성!"):
    st.balloons()
    st.success("참 잘했어요! 정말 먹음직스러운 디저트네요.")
