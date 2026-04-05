import os
import json
import base64
from pathlib import Path
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF (fitz) is not installed. Multimodal PDF image extraction will be disabled. Run `pip install PyMuPDF` to enable it.")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
PROFILE_DIR = DATA_DIR / "profiles"

class DetailedTestResult(BaseModel):
    item_name: str = Field(default="", description="检验项目名称，如'总胆汁酸（TBA）'")
    result: str = Field(default="", description="结果值，如'1.2'")
    unit: str = Field(default="", description="单位，如'# mol/L'")
    reference_range: str = Field(default="", description="参考区间，如'0.0~10.0'")
    abnormal: str = Field(default="", description="是否异常（偏高/偏低/正常），如根据参考值判断")
    record_date: str = Field(default="", description="该化验项目的出具日期，如'2023-10-15'")

class PatientProfile(BaseModel):
    name: str = Field(default="", description="患者姓名（如病历中包含）")
    age: str = Field(default="", description="患者年龄")
    gender: str = Field(default="", description="患者性别")
    record_date: str = Field(default="", description="病历或检查报告的时间（例如：2023-10-15）")
    diagnosis: str = Field(default="", description="主要诊断（例如：鼻咽癌）")
    stage: str = Field(default="", description="肿瘤分期（如：TNM分期、临床分期）")
    treatment_history: str = Field(default="", description="既往治疗史（如化疗、放疗方案及时间）")
    lab_results: str = Field(default="", description="关键检验/病理结果（总体文本描述）")
    test_items: list[DetailedTestResult] = Field(default_factory=list, description="详细的化验指标列表（包含项目名、结果、参考值等）")
    current_status: str = Field(default="", description="当前病情或症状描述")
    medical_summary: str = Field(default="", description="基于上述所有信息的连贯、专业病情总结长文，用于作为大模型的长期记忆")
    suggested_questions: list[str] = Field(default_factory=list, description="根据病历信息，推荐患者向AI医生提问的3-5个个性化问题")

class ProfileManager:
    """患者病历档案管理与多模态提取"""

    def __init__(self):
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.api_key = os.getenv("ARK_API_KEY")
        self.model_name = os.getenv("MODEL")
        self.base_url = os.getenv("BASE_URL")

    def _get_llm(self):
        return init_chat_model(
            model=self.model_name,
            model_provider="openai",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1
        )

    def _extract_images_from_pdf(self, pdf_path: str) -> list[str]:
        """使用 PyMuPDF 将 PDF 每一页渲染为图像，或者直接读取图片，返回 Base64 字符串列表"""
        file_ext = Path(pdf_path).suffix.lower()
        if file_ext in [".jpg", ".jpeg", ".png"]:
            try:
                with open(pdf_path, "rb") as img_file:
                    return [base64.b64encode(img_file.read()).decode("utf-8")]
            except Exception as e:
                print(f"Error reading image: {e}")
                return []

        if not HAS_FITZ:
            return []
        
        base64_images = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                # 调整分辨率 dpi，默认 72，150 足够清晰
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("jpeg")
                b64_str = base64.b64encode(img_data).decode("utf-8")
                base64_images.append(b64_str)
            doc.close()
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
        return base64_images

    def _extract_text_fallback(self, file_path: str) -> str:
        """如果没有 PyMuPDF，回退到使用 Langchain PyPDFLoader 提取纯文本"""
        file_ext = Path(file_path).suffix.lower()
        if file_ext in [".jpg", ".jpeg", ".png"]:
            return "（这是一张图片，但服务器缺少视觉解析库 PyMuPDF，无法提取文字）"
            
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Fallback text extraction failed: {e}")
            return ""

    def process_medical_record(self, user_id: str, file_path: str, filename: str, is_update: bool = False) -> dict:
        """多模态处理病历文件并保存结构化档案"""
        llm = self._get_llm()

        base_prompt = (
            f"请作为一位专业的肿瘤科医生，分析这份上传的病历资料（文件名：{filename}）。"
            "这份资料可能是图片或文本，请提取出患者的核心医疗信息，"
            "并根据病情给出 3-5 个推荐患者向您提问的个性化问题。\n\n"
        )

        if is_update:
            existing_profile = self.load_profile(user_id)
            existing_json = json.dumps(existing_profile, ensure_ascii=False)
            base_prompt += (
                f"【患者历史档案】\n{existing_json}\n\n"
                "【更新指令】\n"
                "这是一份新的病历资料。请你提取这份新资料中的信息，并**将它与上面的历史档案进行融合更新**。\n"
                "1. 对于基本信息（姓名、性别等）若新资料未提及，请保留历史数据。\n"
                "2. **对于详细化验指标 `test_items`，请必须保留历史档案中的全部指标记录，并将新资料中提取到的化验结果追加（append）到该列表中，注意一定要写对每一个指标出具的 `record_date`（化验时间）！**\n"
                "3. **更新长效记忆** `medical_summary`：结合新老病历的数据，总结出病情的发展、治疗的演进和最近的关键变化。\n\n"
            )

        base_prompt += (
            "【重要要求】你必须且只能输出一段合法的 JSON 文本，不要有任何多余的标记（如 ```json）或解释。\n"
            "JSON 必须严格包含以下字段：\n"
            "{\n"
            '  "name": "患者姓名（如病历中包含）",\n'
            '  "age": "患者年龄",\n'
            '  "gender": "患者性别",\n'
            '  "record_date": "最新一次病历或检查报告的时间（例如：2023-10-15）",\n'
            '  "diagnosis": "主要诊断（例如：鼻咽癌）",\n'
            '  "stage": "肿瘤分期（如：TNM分期、临床分期）",\n'
            '  "treatment_history": "既往治疗史（如化疗、放疗方案及时间）",\n'
            '  "lab_results": "关键检验/病理结果（总体文本描述）",\n'
            '  "test_items": [{"item_name": "项目名如总胆红素", "result": "值", "unit": "单位", "reference_range": "参考区间", "abnormal": "异常提示", "record_date": "该化验项目的时间"}],\n'
            '  "current_status": "当前病情或症状描述",\n'
            '  "medical_summary": "请用一段连贯、专业的文字总结上述所有核心病情（如果是更新操作请包含病情演进过程），这将被大模型永久记忆，要求高度概括、准确",\n'
            '  "suggested_questions": ["问题1", "问题2", "问题3"]\n'
            "}\n"
        )

        content_parts = [{"type": "text", "text": base_prompt}]

        if HAS_FITZ:
            images = self._extract_images_from_pdf(file_path)
            if images:
                for b64 in images[:10]: # 限制最多10页以防超过上下文
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
            else:
                # 如果是空图片（可能出错），使用文本回退
                text = self._extract_text_fallback(file_path)
                content_parts.append({"type": "text", "text": f"病历文本内容：\n{text}"})
        else:
            text = self._extract_text_fallback(file_path)
            content_parts.append({"type": "text", "text": f"病历文本内容：\n{text}"})

        messages = [
            SystemMessage(content="你是一个专业的医疗信息提取系统，擅长从病历中提取结构化数据并给出个性化建议。"),
            HumanMessage(content=content_parts)
        ]

        try:
            result = llm.invoke(messages)
            content = result.content.strip()
            
            # 移除可能存在的 markdown 代码块包裹
            if content.startswith("```"):
                content = content.strip("`").replace("json", "", 1).strip()
            
            parsed_json = json.loads(content)
            
            # 使用 Pydantic 验证和补全默认值
            profile_data = PatientProfile(**parsed_json).dict()
        except Exception as e:
            print(f"Medical record extraction error: {e}")
            profile_data = PatientProfile().dict()
            profile_data["current_status"] = f"档案解析失败：{str(e)}"

        # Save profile
        self.save_profile(user_id, profile_data)
        return profile_data

    def save_profile(self, user_id: str, profile_data: dict):
        file_path = PROFILE_DIR / f"{user_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)

    def load_profile(self, user_id: str) -> dict:
        file_path = PROFILE_DIR / f"{user_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
