# Founders' Circle License System: Detailed Plan

## 1. Purpose and Overview

The Founders' Circle License System is designed to manage the pre-launch purchase and post-launch activation of exclusive "Founding Member" subscriptions for Clarity. This system ensures that early adopters receive unique, non-reusable license keys and can seamlessly activate their premium access within Slack after the product's official launch. It integrates payment processing (Polar.sh), backend logic (FastAPI), database management (Supabase), and email delivery for a robust and secure experience.

## 2. Requirements

### General Requirements
*   A user can purchase the Founder plan, receive a unique key via email, and successfully activate their plan in Slack after launch.
*   The key is marked as redeemed upon activation and cannot be reused.
*   The system should be secure, preventing unauthorized key generation or activation.

### Part 1: The Pre-Launch Purchase Flow (User Pays & Receives Key)

**Trigger:** User clicks "Claim Your Founder Spot" on the website and completes payment via Polar.sh.

**Functional Requirements:**
*   **Payment Confirmation:** System must reliably receive `order.succeeded` webhooks from Polar.sh.
*   **Key Generation:** Generate a cryptographically secure, unique license key (UUID-based with a brand prefix).
*   **Database Storage:** Store the generated key, associated customer email, and tier in a dedicated database table.
*   **Email Delivery:** Send a personalized confirmation email containing the license key and activation instructions to the customer.

**Non-Functional Requirements:**
*   **Security:** Webhook verification is mandatory. License keys must be unique and secure.
*   **Reliability:** Key generation and email delivery should be robust against failures.
*   **User Experience:** Confirmation email must be clear, professional, and provide all necessary information.

### Part 2: The Post-Launch Activation Flow (User Activates in Slack)

**Trigger:** A Founding Member installs the Slack bot and attempts to activate their membership.

**Functional Requirements:**
*   **Activation UI:** Provide an intuitive way for users to submit their license key within Slack (App Home button and/or slash command).
*   **Key Validation:** Backend must validate the submitted key against the database (existence, unredeemed status).
*   **Subscription Upgrade:** Upon successful validation, update the user's workspace subscription to the "Founder" tier.
*   **Key Redemption:** Mark the license key as redeemed in the database, linking it to the activating workspace and user.
*   **In-Slack Confirmation:** Send a clear confirmation message to the user in Slack upon successful activation.

**Non-Functional Requirements:**
*   **Security:** Key validation must prevent reuse and unauthorized activation.
*   **User Experience:** Activation process should be straightforward with clear feedback for success or failure.
*   **Data Integrity:** Database updates must be atomic and consistent.

## 3. Explanation of the Flow

### Part 1: The Pre-Launch Purchase Flow

1.  **User Purchase (Website -> Polar.sh):** A user navigates to `claritypm.xyz`, clicks "Claim Your Founder Spot," and completes the payment process on Polar.sh.
2.  **Polar Webhook (`Polar.sh` -> `FastAPI Backend`):** Upon successful payment, Polar.sh automatically sends an `order.succeeded` webhook to a dedicated endpoint on the Clarity FastAPI backend (`/webhooks/polar-license`). This webhook contains crucial information like the customer's email and the purchased product details.
3.  **Key Generation & Storage (`FastAPI Backend` -> `Supabase`):**
    *   The FastAPI endpoint receives and verifies the webhook.
    *   It generates a unique, cryptographically secure license key (e.g., `CLARITY-FOUNDER-a1b2c3d4-e5f6-7890-1234-56789abcdef0`).
    *   This key, along with the `customer_email`, `tier` ("Founder"), and an `is_redeemed: false` status, is securely stored in the `license_keys` table in the Supabase database.
4.  **Confirmation Email (`FastAPI Backend` -> `Email Service` -> `Customer`):**
    *   The backend triggers a transactional email (via SendGrid or similar service) to the `customer_email`.
    *   The email includes a warm welcome, the unique license key, instructions for post-launch activation, and a link to the private Founders' Slack community.

### Part 2: The Post-Launch Activation Flow

1.  **User Onboarding (Slack):** After Clarity's official launch, a Founding Member installs the Slack bot into their workspace. This creates initial workspace and user records in the Supabase database.
2.  **Activation UI (Slack):**
    *   **App Home:** A prominent "Activate Founder Membership" button is displayed in the Clarity App Home within Slack.
    *   **Slash Command:** Alternatively, the user can type `/clarity-activate` in any Slack channel or DM.
    *   Both methods open a Slack modal (using Block Kit) that prompts the user to paste their unique digital license key.
3.  **Key Validation & Activation (`Slack` -> `FastAPI Backend` -> `Supabase`):**
    *   The submitted license key, along with the `workspace_id` and `user_id` of the activating user, is sent to a FastAPI activation endpoint (`/slack/activate-license`).
    *   The backend queries the `license_keys` table to:
        *   Confirm the `license_key` exists.
        *   Verify that `is_redeemed` is `false`.
    *   If valid and unredeemed:
        *   The `workspace_subscriptions` table is updated for the `workspace_id`, setting the subscription plan to "Founder."
        *   The `license_keys` table is updated: `is_redeemed` is set to `true`, and `redeemed_by_workspace_id` and `redeemed_by_user_id` are populated.
4.  **Confirmation (Slack):** The bot sends a direct confirmation message to the user in Slack: "Welcome, Founder! Your exclusive membership has been activated. Thank you for being one of our first supporters."

## 4. Action Plan to Implement

### Phase 1: Pre-Launch Purchase Flow

1.  **Create `license_keys` Database Table (Migration)**
    *   **File**: `migrations/0008_create_license_keys_table.sql`
    *   **Content**:
        ```sql
        CREATE TABLE IF NOT EXISTS public.license_keys (
            id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
            license_key TEXT UNIQUE NOT NULL,
            customer_email TEXT NOT NULL,
            tier TEXT NOT NULL,
            is_redeemed BOOLEAN DEFAULT FALSE NOT NULL,
            redeemed_by_workspace_id TEXT NULL,
            redeemed_by_user_id TEXT NULL,
            client_reference_id TEXT NULL, -- NEW: Link to founding_members
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            redeemed_at TIMESTAMP WITH TIME ZONE NULL
        );

        ALTER TABLE public.license_keys ENABLE ROW LEVEL SECURITY;

        CREATE POLICY "Enable service role to manage all license keys"
        ON public.license_keys
        FOR ALL
        USING (true) WITH CHECK (true);
        ```
    *   **Dependencies**: `uuid-ossp` extension in Supabase.

2.  **Develop Netlify Function for Polar Webhooks**
    *   **File**: `netlify/functions/polar-license-webhook.js`
    *   **Endpoint**: `/.netlify/functions/polar-license-webhook`
    *   **Logic**:
        *   Receive `order.succeeded` webhooks from Polar.
        *   Implement webhook signature verification using `polar-sdk`'s Node.js webhook verification.
        *   Extract `customer_email`, `product_name` (tier), and **`client_reference_id`** from the webhook payload.
        *   Generate `license_key = \`CLARITY-FOUNDER-${uuid.v4()}\``.
        *   Insert into `license_keys` table: `supabase.from('license_keys').insert({... , client_reference_id: client_reference_id}).execute()`.
        *   Trigger a transactional email via SendGrid to the `customer_email` with the license key, activation instructions, and Slack community link.
        *   Return a `200 OK` response to Polar.

### Phase 2: Post-Launch Activation Flow

1.  **Design In-Slack UI for License Activation**
    *   **File**: `main.py`
    *   **App Home Button**:
        *   Modify `slack_app.event("app_home_opened")` listener.
        *   Check `workspace_subscriptions` for the `team_id`.
        *   If not "Founder" tier, use `client.views_publish` with a Block Kit payload to display an "Activate Founder Membership" button.
        *   The button will trigger a Slack modal (e.g., `view_submission` with `callback_id='activate_license_modal'`).
    *   **Slash Command**:
        *   Register `/clarity-activate` in Slack App settings.
        *   Add `slack_app.command("/clarity-activate")` listener.
        *   On invocation, use `client.views_open` with a Block Kit modal payload.
        *   Modal content: Input field for license key, submit button.

2.  **Develop FastAPI Endpoint for License Activation**
    *   **File**: `main.py`
    *   **Endpoint**: `slack_app.view("activate_license_modal")` listener (for modal submission).
    *   **Logic**:
        *   Extract `license_key` from the modal submission payload.
        *   Extract `workspace_id` and `user_id` from the Slack context.
        *   Query `license_keys` table: `rls_supabase_client.from_('license_keys').select('*').eq('license_key', license_key).single().execute()`.
        *   Validate: Check if key exists and `is_redeemed` is `false`.
        *   If valid:
            *   Update `workspace_subscriptions`: `rls_supabase_client.from_('workspace_subscriptions').update({'plan_id': license_data['tier']}).eq('workspace_id', workspace_id).execute()`.
            *   Update `license_keys`: `rls_supabase_client.from_('license_keys').update({'is_redeemed': true, 'redeemed_by_workspace_id': workspace_id, 'redeemed_by_user_id': user_id, 'redeemed_at': datetime.now(timezone.utc)}).eq('license_key', license_key).execute()`.
            *   Send confirmation message to user via `client.chat_postMessage`.
        *   If invalid/redeemed: Send an error message to the user in Slack.

3.  **Implement Robust Error Handling for License System**
    *   **File**: `main.py` and `netlify/functions/polar-license-webhook.js`
    *   **Logic**:
        *   Use `try-except` blocks for all database and external API calls.
        *   Log detailed errors using `logger.error()` or `console.error()`.
        *   Provide user-friendly error messages in Slack (for activation flow) or HTTP responses (for webhooks).
        *   Consider specific error codes for different failure types (e.g., `400 Bad Request` for invalid key, `409 Conflict` for redeemed key).

## 5. Dependencies and Environment Variables

*   **New Python Libraries**: `uuid` (standard library).
*   **New JavaScript Libraries (for Netlify Function)**: `uuid` (already in `package.json`).
*   **New Supabase Extension**: `uuid-ossp` (if not already enabled).
*   **New Environment Variables**:
    *   `POLAR_WEBHOOK_SECRET`: For verifying Polar webhooks.
    *   `SENDGRID_API_KEY`: For sending emails from Netlify Function.
    *   `SENDER_EMAIL`: For sending emails from Netlify Function.
    *   `SLACK_FOUNDER_COMMUNITY_LINK`: URL for the private Slack community.

## 6. Additional Actions (Addressing Typo Scenario)

1.  **Add `client_reference_id` to `license_keys` table (Migration)**
    *   **File**: `migrations/0009_add_client_ref_id_to_license_keys.sql`
    *   **Content**:
        ```sql
        ALTER TABLE public.license_keys
        ADD COLUMN client_reference_id TEXT NULL;

        -- Optional: Add an index for faster lookups if this will be frequently queried
        CREATE INDEX IF NOT EXISTS idx_license_keys_client_reference_id
        ON public.license_keys (client_reference_id);
        ```
    *   **Purpose**: To link the generated license key to the original pre-payment user data, enabling support to recover keys even if the user made a typo in their email on Polar.sh.

2.  **Update `netlify/functions/polar-license-webhook.js` to store `client_reference_id`**
    *   **File**: `netlify/functions/polar-license-webhook.js`
    *   **Logic**:
        *   Extract `client_reference_id` from the Polar webhook payload (e.g., `webhookEvent.data.client_reference_id`).
        *   Include `client_reference_id: clientReferenceId` in the Supabase insert operation for the `license_keys` table.
    *   **Purpose**: To ensure the `client_reference_id` is persisted with the license key, enabling the recovery mechanism.

## 5. Dependencies and Environment Variables

*   **New Python Libraries**: `uuid` (standard library), `sendgrid` (if not already used for Netlify functions).
*   **New Supabase Extension**: `uuid-ossp` (if not already enabled).
*   **New Environment Variables**:
    *   `POLAR_WEBHOOK_SECRET`: For verifying Polar webhooks.
    *   `SENDGRID_API_KEY`: If using SendGrid directly from FastAPI.
    *   `SLACK_FOUNDER_COMMUNITY_LINK`: URL for the private Slack community.
