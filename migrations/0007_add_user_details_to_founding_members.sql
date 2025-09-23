-- Add new columns to the founding_members table for CRM and tracking purposes

-- Add name and surname
ALTER TABLE "public"."founding_members"
ADD COLUMN "name" text,
ADD COLUMN "surname" text;

-- Add phone and country
ALTER TABLE "public"."founding_members"
ADD COLUMN "phone" text,
ADD COLUMN "country" text;

-- Add a status column to track the payment flow
ALTER TABLE "public"."founding_members"
ADD COLUMN "payment_status" text DEFAULT 'initiated';

-- Add a timestamp for when the record was created
ALTER TABLE "public"."founding_members"
ADD COLUMN "created_at" timestamp with time zone DEFAULT now() NOT NULL;

-- Add a timestamp for when the record was last updated
ALTER TABLE "public"."founding_members"
ADD COLUMN "updated_at" timestamp with time zone DEFAULT now() NOT NULL;

-- Make email unique to prevent duplicate entries
ALTER TABLE "public"."founding_members"
ADD CONSTRAINT "founding_members_email_key" UNIQUE ("email");
