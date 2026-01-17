import streamlit as st
import os
from PIL import Image, ImageChops

# 페이지 설정
st.set_page_config(page_title="달콤한 디저트 공방", page_icon="🍰")

def apply_color_to_image(image_path, color_hex):
    """이미지의 밝은 부분에 사용자가 선택한 색상을 입히는 함수"""
    try:
        # 이미지 불러오기 및 RGBA 변환
        img = Image.open(image_path).convert("RGBA")
        
        # Hex 색상을 RGB로 변환
        color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # 선택한 색상의 단색 레이어 생성
        color_layer = Image.new("RGBA", img.size, color_rgb + (255,))
        
        # Multiply(승산) 합성을 통해 색상 적용 (이미지의 질감 유지)
        colored_img = ImageChops.multiply(img, color_layer)
        
        # 원본의 알파 채널(투명도) 유지
        img.paste(colored_img, (0, 0), img)
        return img
    except Exception as e:
        return None

# 메인 타이틀
st.title("🎨 디저트 색깔 입히기 공방")

# 사이드바 설정
st.sidebar.header("🛠️ 꾸미기 도구함")

dessert_options = {
    "아이스크림": "icecream.png",
    "빙수": "shaved_ice.png",
    "젤리빈": "jellybean.png",
    "마카롱": "macaron.png",
    "쿠키": "cookie.png"
}

selected_name = st.sidebar.selectbox("디저트 선택", list(dessert_options.keys()))
chosen_color = st.sidebar.color_picker("디저트에 입힐 색상 선택", "#FFB6C1")

# 메인 화면
st.divider()

image_filename = dessert_options[selected_name]
image_path = os.path.join("assets", image_filename)

if os.path.exists(image_path):
    # 색상 적용 로직 실행
    final_dessert = apply_color_to_image(image_path, chosen_color)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if final_dessert:
            st.image(final_dessert, caption=f"내가 만든 {chosen_color}색 {selected_name}", use_container_width=True)
    
    with col2:
        st.write("### 📝 디자인 정보")
        st.success(f"현재 선택: **{selected_name}**")
        st.write(f"색상 코드: `{chosen_color}`")
        
        # 결과물 다운로드 버튼
        import io
        buf = io.BytesIO()
        final_dessert.save(buf, format="PNG")
        st.download_button("🖼️ 이미지 저장하기", buf.getvalue(), f"my_{selected_name}.png", "image/png")

else:
    st.error(f"assets 폴더에 '{image_filename}' 파일이 없습니다.")

# 오류가 났던 부분 수정: st.confetti() 삭제 후 st.balloons() 사용
if st.button("🎉 완성!"):
    st.balloons()
    st.success("정말 예쁜 디저트네요!")
