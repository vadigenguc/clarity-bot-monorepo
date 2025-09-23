-- This script creates the founding_members table with all necessary columns.

CREATE TABLE "public"."founding_members" (
    "id" uuid DEFAULT gen_random_uuid() NOT NULL,
    "email" text NOT NULL,
    "polar_customer_id" text,
    "polar_subscription_id" text,
    "name" text,
    "surname" text,
    "phone" text,
    "country" text,
    "payment_status" text DEFAULT 'initiated',
    "created_at" timestamp with time zone DEFAULT now() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT now() NOT NULL,
    PRIMARY KEY ("id"),
    UNIQUE ("email")
);
