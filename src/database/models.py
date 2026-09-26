from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Role(str, Enum):
    FARMER = "farmer"
    BUYER = "buyer"


class ListingStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESERVED = "RESERVED"
    SOLD = "SOLD"
    CANCELLED = "CANCELLED"


class RequestStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class OrderStatus(str, Enum):
    ORDER_PLACED = "ORDER_PLACED"
    SELLER_CONFIRMED = "SELLER_CONFIRMED"
    PICKUP_SCHEDULED = "PICKUP_SCHEDULED"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    phone: Mapped[str | None] = mapped_column(String(30))
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    location: Mapped["UserLocation | None"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserLocation(Base):
    __tablename__ = "user_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    address: Mapped[str | None] = mapped_column(String(255))
    village: Mapped[str | None] = mapped_column(String(120))
    taluk: Mapped[str | None] = mapped_column(String(120))
    district: Mapped[str | None] = mapped_column(String(120))
    pincode: Mapped[str | None] = mapped_column(String(10))
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))

    user: Mapped[User] = relationship(back_populates="location")


class FarmerListing(Base):
    __tablename__ = "farmer_listings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    farmer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    crop: Mapped[str] = mapped_column(String(80), nullable=False)
    variety: Mapped[str] = mapped_column(String(120), nullable=False)
    grade: Mapped[str] = mapped_column(String(80), nullable=False)
    quantity_quintals: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    minimum_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    market: Mapped[str] = mapped_column(String(120), nullable=False)
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    available_from: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=ListingStatus.ACTIVE.value, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    farmer: Mapped[User] = relationship()


class BuyerRequest(Base):
    __tablename__ = "buyer_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    buyer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    farmer_listing_id: Mapped[int | None] = mapped_column(ForeignKey("farmer_listings.id"), index=True)
    crop: Mapped[str] = mapped_column(String(80), nullable=False)
    variety: Mapped[str] = mapped_column(String(120), nullable=False)
    grade: Mapped[str] = mapped_column(String(80), nullable=False)
    quantity_quintals: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    offered_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    preferred_market: Mapped[str | None] = mapped_column(String(120))
    pickup_location: Mapped[str | None] = mapped_column(String(255))
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    required_by_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default=RequestStatus.PENDING.value, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    buyer: Mapped[User] = relationship()
    farmer_listing: Mapped[FarmerListing | None] = relationship()


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    farmer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    buyer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    farmer_listing_id: Mapped[int] = mapped_column(ForeignKey("farmer_listings.id"), nullable=False)
    buyer_request_id: Mapped[int] = mapped_column(ForeignKey("buyer_requests.id"), nullable=False)
    quantity_quintals: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    agreed_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    pickup_location: Mapped[str | None] = mapped_column(String(255))
    delivery_location: Mapped[str | None] = mapped_column(String(255))
    order_date: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    estimated_pickup_date: Mapped[date] = mapped_column(Date, nullable=False)
    estimated_delivery_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default=OrderStatus.ORDER_PLACED.value, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    farmer: Mapped[User] = relationship(foreign_keys=[farmer_id])
    buyer: Mapped[User] = relationship(foreign_keys=[buyer_id])
    listing: Mapped[FarmerListing] = relationship()
    request: Mapped[BuyerRequest] = relationship()
    messages: Mapped[list["Message"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True, nullable=False)
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    receiver_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    order: Mapped[Order] = relationship(back_populates="messages")
    sender: Mapped[User] = relationship(foreign_keys=[sender_id])
    receiver: Mapped[User] = relationship(foreign_keys=[receiver_id])


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    district: Mapped[str] = mapped_column(String(120), nullable=False)
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    coordinate_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    address: Mapped[str | None] = mapped_column(String(255))
