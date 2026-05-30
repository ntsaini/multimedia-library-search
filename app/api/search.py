from fastapi import APIRouter, File, UploadFile

from app.services.search_service import search_by_reference_image_bytes, search_person_by_name
router = APIRouter()


@router.get("/api/search")
def search_by_name(name: str = ""):
    return search_person_by_name(name=name, limit=200)


@router.post("/api/search/photo")
async def search_by_photo(file: UploadFile = File(...)):
    data = await file.read()
    return search_by_reference_image_bytes(data, distance_threshold=0.5, limit=100)
