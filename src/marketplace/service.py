from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from math import ceil

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import BuyerRequest, FarmerListing, ListingStatus, Message, Order, OrderStatus, RequestStatus, User


@dataclass(frozen=True)
class MatchCandidate:
    listing: FarmerListing
    farmer: User
    distance_km: float | None
    available_quantity: Decimal
    quantity_requested: Decimal
    price_compatible: bool


def create_farmer_listing(session: Session, *, farmer_id: int, crop: str, variety: str, grade: str, quantity_quintals: Decimal, minimum_price: Decimal, market: str, latitude: Decimal | None, longitude: Decimal | None, available_from: date) -> FarmerListing:
    if quantity_quintals <= 0 or minimum_price < 0:
        raise ValueError("Quantity must be positive and price cannot be negative.")
    listing = FarmerListing(farmer_id=farmer_id, crop=crop, variety=variety, grade=grade, quantity_quintals=quantity_quintals, minimum_price=minimum_price, market=market, latitude=latitude, longitude=longitude, available_from=available_from)
    session.add(listing)
    session.flush()
    return listing


def create_buyer_request(session: Session, *, buyer_id: int, crop: str, variety: str, grade: str, quantity_quintals: Decimal, offered_price: Decimal, preferred_market: str | None, pickup_location: str | None, latitude: Decimal | None, longitude: Decimal | None, required_by_date: date | None, farmer_listing_id: int | None = None) -> BuyerRequest:
    if quantity_quintals <= 0 or offered_price < 0:
        raise ValueError("Quantity must be positive and offer price cannot be negative.")
    request = BuyerRequest(buyer_id=buyer_id, farmer_listing_id=farmer_listing_id, crop=crop, variety=variety, grade=grade, quantity_quintals=quantity_quintals, offered_price=offered_price, preferred_market=preferred_market, pickup_location=pickup_location, latitude=latitude, longitude=longitude, required_by_date=required_by_date)
    session.add(request)
    session.flush()
    return request


def find_live_listings(session: Session, *, crop: str, variety: str, grade: str) -> list[FarmerListing]:
    return list(session.scalars(select(FarmerListing).where(FarmerListing.status == ListingStatus.ACTIVE.value, FarmerListing.crop == crop, FarmerListing.variety == variety, FarmerListing.grade == grade).order_by(FarmerListing.created_at.desc())).all())


def _distance(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    try:
        from src.market_comparison import calculate_haversine_distance
        return calculate_haversine_distance(float(lat1), float(lon1), float(lat2), float(lon2))
    except ModuleNotFoundError as exc:
        if exc.name != "src.market_comparison":
            raise
        # Isolated unit-test fallback; the application uses the project's
        # existing market_comparison implementation when it is available.
        from math import asin, cos, radians, sin, sqrt
        r = 6371.0
        dlat = radians(float(lat2) - float(lat1))
        dlon = radians(float(lon2) - float(lon1))
        a = sin(dlat / 2) ** 2 + cos(radians(float(lat1))) * cos(radians(float(lat2))) * sin(dlon / 2) ** 2
        return 2 * r * asin(sqrt(a))


def match_buyer_to_listings(session: Session, request: BuyerRequest) -> list[MatchCandidate]:
    listings = find_live_listings(session, crop=request.crop, variety=request.variety, grade=request.grade)
    buyer = request.buyer
    buyer_location = buyer.location if buyer else None
    candidates = []
    for listing in listings:
        distance = _distance(buyer_location.latitude if buyer_location else None, buyer_location.longitude if buyer_location else None, listing.latitude, listing.longitude)
        candidates.append(MatchCandidate(listing=listing, farmer=listing.farmer, distance_km=distance, available_quantity=Decimal(listing.quantity_quintals), quantity_requested=Decimal(request.quantity_quintals), price_compatible=Decimal(request.offered_price) >= Decimal(listing.minimum_price)))
    return sorted(candidates, key=lambda x: (not x.price_compatible, x.distance_km is None, x.distance_km or 10**9, -float(x.listing.minimum_price)))


def request_purchase(session: Session, *, buyer_id: int, listing_id: int, quantity_quintals: Decimal, offered_price: Decimal, pickup_location: str | None, required_by_date: date | None) -> BuyerRequest:
    listing = session.get(FarmerListing, listing_id)
    if listing is None or listing.status != ListingStatus.ACTIVE.value:
        raise ValueError("The selected farmer listing is no longer active.")
    if quantity_quintals <= 0 or quantity_quintals > Decimal(listing.quantity_quintals):
        raise ValueError(f"Requested quantity cannot exceed the farmer's available quantity of {listing.quantity_quintals} quintals.")
    buyer = session.get(User, buyer_id)
    request = create_buyer_request(session, buyer_id=buyer_id, farmer_listing_id=listing.id, crop=listing.crop, variety=listing.variety, grade=listing.grade, quantity_quintals=quantity_quintals, offered_price=offered_price, preferred_market=listing.market, pickup_location=pickup_location, latitude=buyer.location.latitude if buyer and buyer.location else None, longitude=buyer.location.longitude if buyer and buyer.location else None, required_by_date=required_by_date)
    return request


def _estimated_dates(order_date: datetime, quantity: Decimal, distance_km: float | None) -> tuple[date, date]:
    readiness_days = 1 if quantity <= 50 else 2 if quantity <= 100 else 3
    distance_days = 1 if distance_km is None else max(1, ceil(distance_km / 100))
    pickup = (order_date + timedelta(days=readiness_days)).date()
    delivery = pickup + timedelta(days=distance_days)
    return pickup, delivery


def accept_request(session: Session, *, request_id: int, farmer_id: int) -> Order:
    request = session.get(BuyerRequest, request_id)
    if request is None or request.status != RequestStatus.PENDING.value:
        raise ValueError("Purchase request is not pending.")
    listing = request.farmer_listing
    if listing is None or listing.farmer_id != farmer_id or listing.status != ListingStatus.ACTIVE.value:
        raise ValueError("You are not authorized to accept this request or the listing is unavailable.")
    if Decimal(request.quantity_quintals) > Decimal(listing.quantity_quintals):
        raise ValueError("The requested quantity is no longer available.")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    distance = _distance(request.latitude, request.longitude, listing.latitude, listing.longitude)
    pickup, delivery = _estimated_dates(now, Decimal(request.quantity_quintals), distance)
    total = Decimal(request.quantity_quintals) * Decimal(request.offered_price)
    order = Order(farmer_id=listing.farmer_id, buyer_id=request.buyer_id, farmer_listing_id=listing.id, buyer_request_id=request.id, quantity_quintals=request.quantity_quintals, agreed_price=request.offered_price, total_amount=total, pickup_location=request.pickup_location, delivery_location=listing.market, order_date=now, estimated_pickup_date=pickup, estimated_delivery_date=delivery, status=OrderStatus.SELLER_CONFIRMED.value)
    request.status = RequestStatus.ACCEPTED.value
    remaining = Decimal(listing.quantity_quintals) - Decimal(request.quantity_quintals)
    listing.quantity_quintals = remaining
    if remaining == 0:
        listing.status = ListingStatus.SOLD.value
    session.add(order)
    session.flush()
    return order


def reject_request(session: Session, *, request_id: int, farmer_id: int) -> BuyerRequest:
    request = session.get(BuyerRequest, request_id)
    if request is None or request.farmer_listing is None or request.farmer_listing.farmer_id != farmer_id:
        raise ValueError("Purchase request not found for this farmer.")
    request.status = RequestStatus.REJECTED.value
    session.flush()
    return request


def advance_order(session: Session, *, order_id: int, user_id: int, next_status: str) -> Order:
    order = session.get(Order, order_id)
    if order is None or user_id not in {order.farmer_id, order.buyer_id}:
        raise ValueError("Order not found for this user.")
    allowed = {
        OrderStatus.ORDER_PLACED.value: {OrderStatus.SELLER_CONFIRMED.value},
        OrderStatus.SELLER_CONFIRMED.value: {OrderStatus.PICKUP_SCHEDULED.value},
        OrderStatus.PICKUP_SCHEDULED.value: {OrderStatus.IN_TRANSIT.value},
        OrderStatus.IN_TRANSIT.value: {OrderStatus.DELIVERED.value},
    }
    if next_status not in allowed.get(order.status, set()):
        raise ValueError(f"Cannot move {order.status} to {next_status}.")
    order.status = next_status
    session.flush()
    return order


def send_message(session: Session, *, order_id: int, sender_id: int, message: str) -> Message:
    order = session.get(Order, order_id)
    if order is None or sender_id not in {order.farmer_id, order.buyer_id}:
        raise ValueError("You cannot message participants of this order.")
    receiver_id = order.buyer_id if sender_id == order.farmer_id else order.farmer_id
    clean = message.strip()
    if not clean:
        raise ValueError("Message cannot be empty.")
    row = Message(order_id=order_id, sender_id=sender_id, receiver_id=receiver_id, message=clean)
    session.add(row)
    session.flush()
    return row


def order_history(session: Session, user_id: int) -> list[Order]:
    return list(session.scalars(select(Order).where((Order.farmer_id == user_id) | (Order.buyer_id == user_id)).order_by(Order.created_at.desc())).all())
