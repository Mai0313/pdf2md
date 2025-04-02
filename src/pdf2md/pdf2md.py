from typing import Any
import asyncio
from pathlib import Path
import warnings

import httpx
from openai import AsyncAzureOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, AliasChoices, computed_field
from marker.models import create_model_dict
from marker.output import save_output
from pydantic_settings import BaseSettings
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri

warnings.filterwarnings("ignore", category=ResourceWarning)


class ExtractedImage(BaseModel):
    idx: int
    image_url: Path
    description: str = Field(default="")

    @computed_field
    @property
    def description_result(self) -> str:
        description_result = (
            f"Here is the image description of {self.image_url}:\n```\n{self.description}\n```"
        )
        return description_result


class DocsConverter(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: str = Field(
        default="./docs",
        description="The path of the docs you want to convert, it can be either a file or a directory.",
        frozen=False,
        deprecated=False,
    )
    max_processes: int = Field(
        default=10,
        description="The maximum number of processes to use for conversion.",
        frozen=False,
        deprecated=False,
    )
    user_id: str = Field(
        default="srv_dvc_tma001",
        description="The User ID for the LLM API request.",
        examples=["srv_dvc_tma001", "ds906659"],
        frozen=False,
        validation_alias=AliasChoices("USER_ID"),
        serialization_alias="user_id",
    )
    api_key: str = Field(
        ...,
        description="The API key for the LLM API request.",
        examples=["eyJh..."],
        frozen=False,
        validation_alias=AliasChoices("API_KEY"),
        serialization_alias="api_key",
    )
    base_url: str = Field(
        default="https://mlop-azure-gateway.mediatek.inc",
        description="The base URL for the LLM API request.",
        examples=[
            "https://mlop-azure-gateway.mediatek.inc",  # OA
            "https://mtklm-oa.mediatek.inc/llm/api/v3/models",  # OA CSES Playground
            "https://mlop-gateway-hwrd.mediatek.inc",  # HWRD
            "https://mtklm-hwrd.mediatek.inc/llm/api/v3/models",  # HWRD CSES Playground
        ],
        frozen=False,
        validation_alias=AliasChoices("BASE_URL"),
        serialization_alias="base_url",
    )

    @computed_field
    @property
    def all_docs_paths(self) -> list[Path]:
        if Path(self.path).is_dir():
            all_docs_paths = list(Path(self.path).rglob("*"))
        elif Path(self.path).is_file():
            all_docs_paths = [Path(self.path)]
        else:
            raise ValueError(f"Invalid path: {self.path}")
        return all_docs_paths

    @computed_field
    @property
    def semaphore(self) -> asyncio.Semaphore:
        semaphore = asyncio.Semaphore(value=self.max_processes)
        return semaphore

    @computed_field
    @property
    def client(self) -> AsyncAzureOpenAI:
        client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,
            api_version="2024-12-01-preview",
            http_client=httpx.AsyncClient(headers={"X-User-Id": self.user_id}),
        )
        return client

    async def __process_image(self, extracted_image: ExtractedImage) -> ExtractedImage:
        # 使用 semaphore 控制並發數量
        async with self.semaphore:
            # https://platform.openai.com/docs/guides/vision
            resolved_path = extracted_image.image_url.resolve().as_posix()
            if extracted_image.image_url.exists():
                logfire.info("Processing Image...", image_path=resolved_path)
                # 如果圖片路徑存在，讀取圖片並轉換成 data uri 格式
                base64_image = get_pil_image(image_file=resolved_path)
                image_uri = pil_to_data_uri(base64_image)
                content: list[dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": "Describe the image in detail."},
                ]
                # 呼叫 API 取得描述
                response = await self.client.chat.completions.create(
                    model="aide-gpt-4o",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                )
                result = response.choices[0].message.content
                if not result:
                    result = f"The Image Description is Empty: {resolved_path}."
                    logfire.error(result)
            else:
                result = f"Failed to Find the Image: {resolved_path}."
                logfire.error(result)
            extracted_image.description = result
            return extracted_image

    async def to_markdown(self) -> None:
        all_docs_paths = [
            f for f in self.all_docs_paths if f.suffix == ".pdf" or f.suffix == ".txt"
        ]

        if not all_docs_paths:
            logfire.warn("No pdf files found in the path.")
            return

        config_parser = ConfigParser(
            cli_options={"languages": "en", "output_format": "markdown", "output_dir": "parsed"}
        )

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        for docs_path in all_docs_paths:
            output_dir = docs_path.with_suffix("")
            output_dir = output_dir.with_name(output_dir.name.replace(" ", "_"))
            if output_dir.is_dir() and output_dir.exists():
                logfire.info("Skip existing dir", source=docs_path, output=output_dir.as_posix())
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            if docs_path.suffix == ".pdf":
                logfire.info("Converting PDF...", source=docs_path)
                rendered = converter(filepath=docs_path.as_posix())
                parsed_name = docs_path.stem.replace(" ", "_")
                save_output(
                    rendered=rendered, output_dir=output_dir.as_posix(), fname_base=parsed_name
                )
            elif docs_path.suffix == ".txt":
                logfire.info("Converting TXT...", source=docs_path)
                content = docs_path.read_text(encoding="utf-8")
                # 直接將內容寫入 markdown 檔案
                parsed_name = docs_path.stem.replace(" ", "_")
                output_file = output_dir / f"{parsed_name}.md"
                output_file.write_text(content, encoding="utf-8")
            logfire.info("Converted Successfully", source=docs_path, output=output_dir.as_posix())

    async def parse_docs_with_images(self) -> None:
        docs_paths = [f for f in self.all_docs_paths if f.name.endswith(".md")]
        docs_paths = [f for f in docs_paths if not f.stem.endswith("_parsed")]
        for docs_path in docs_paths:
            docs_contents = docs_path.read_text(encoding="utf-8").splitlines()
            extracted_images: list[ExtractedImage] = []
            for idx, line in enumerate(docs_contents, start=1):
                if line.startswith("![](_page") and line.endswith(")"):
                    image_path_string = line.split("](")[1].split(")")[0]
                    image_path = docs_path.parent / image_path_string
                    if image_path.exists():
                        extracted_images.append(
                            ExtractedImage(
                                idx=idx, image_url=image_path.absolute(), description=""
                            )
                        )
            # For debugging
            # extract_images = extract_images[:1]
            tasks = [self.__process_image(extracted_image) for extracted_image in extracted_images]
            parsed_images = await asyncio.gather(*tasks)
            for parsed_image in parsed_images:
                docs_contents[parsed_image.idx - 1] = parsed_image.description_result
            parsed_content = "\n".join(docs_contents)
            new_docs_path = docs_path.with_name(f"{docs_path.stem}_parsed{docs_path.suffix}")
            new_docs_path.write_text(parsed_content, encoding="utf-8")
            logfire.info("Parsed Successfully", source=docs_path, output=new_docs_path.as_posix())

    async def __call__(self) -> None:
        await self.to_markdown()
        await self.parse_docs_with_images()


if __name__ == "__main__":
    import asyncio

    converter = DocsConverter(path="./docs/pdfs")
    asyncio.run(converter.to_markdown())
    # asyncio.run(converter.parse_docs_with_images())
