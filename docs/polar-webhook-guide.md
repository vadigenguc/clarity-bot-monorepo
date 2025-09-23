# Setup Webhooks

> Get notifications asynchronously when events occur instead of having to poll for updates

Our webhook implementation follows the [Standard Webhooks](https://www.standardwebhooks.com/) specification
and our SDKs offer:

* Built-in webhook signature validation for security
* Fully typed webhook payloads

In addition, our webhooks offer built-in support for **Slack** & **Discord**
formatting. Making it a breeze to setup in-chat notifications for your team.

## Get Started

<Info>
  **Use our sandbox environment during development**

  So you can easily test purchases, subscriptions, cancellations and refunds to
  automatically trigger webhook events without spending a dime.
</Info>

<Steps>
  <Step title="Add new endpoint">
    Head over to your organization settings and click on the `Add Endpoint` button to create a new webhook.

    <img className="block dark:hidden" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=23eb5510c1dbd2f511461dfe1e262485" width="1532" height="389" data-path="assets/integrate/webhooks/create.light.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=a351c94bc7d002cfe01d4b90d8ec583f 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=bf3519e1ee9851a86af38687e0f485fc 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=1da4ac8731e29d4854dfc874051725c0 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=1a925a887349c660469c221438ce8472 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=0e7d63bd8ed6e7d29c84f6f31d376042 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.light.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=989ec443952fc397adca0fcd68e2ab8d 2500w" data-optimize="true" data-opv="2" />

    <img className="hidden dark:block" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=c86d0770b5dc4d42279d9f1da568edb8" width="1494" height="388" data-path="assets/integrate/webhooks/create.dark.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=32ccd01674a4936933650414fc920523 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=642a4166857e8c82092f035c5055b30c 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=f5dc475d503ecbb05ee41e6c879756f8 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=1421996da6c06e3cfe3ad7212579e3c4 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=ac70b9d0291437b5073e22fe413bee30 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/create.dark.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=adae7e7ef7b875a18a6582b7b7c2f3c0 2500w" data-optimize="true" data-opv="2" />
  </Step>

  <Step title="Specify your endpoint URL">
    Enter the URL to which the webhook events should be sent.

    <img className="block dark:hidden" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=7ef69a33df9105e42fd8ab618a30669d" width="1075" height="402" data-path="assets/integrate/webhooks/url.light.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=9b83d589c935a1114d1abe2363ccc320 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=7e07c55097be4d033521f8df4cc3d10d 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=503ebb430cff99a14ec5a6958479ddfd 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=2f13c432330d46fd247bb7865083c8c5 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=65760df05d7310fa490cc0b7714dd7fc 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.light.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=806064df619d5c8d9bae4a61688310ad 2500w" data-optimize="true" data-opv="2" />

    <img className="hidden dark:block" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=6a4514f5bd12fcf01ab1aa1bc2b1f26a" width="1068" height="398" data-path="assets/integrate/webhooks/url.dark.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=a2a279ce6d5955c0cb1f79504e880cb1 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=aff292b704a653df95e8f7a0fdbf6c62 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=e05077418ae849f841b8817725095f7b 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=11a85bb4ca26112804fafa276290e53a 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=ff125ccdf3b378753f5022621b0b78a8 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/url.dark.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=ec6538e59e41c92123760a599197ade0 2500w" data-optimize="true" data-opv="2" />

    <Tip>
      **Developing locally?**

      Use a tool like [ngrok](https://ngrok.com/) to tunnel webhook events to your local development environment. This will allow you to test your webhook handlers without deploying them to a live server.

      Once you have `ngrok` you can easily start a tunnel:

      ```bash
      ngrok http 3000
      ```

      Just be sure to provide the URL ngrok gives you as the webhook endpoint on
      Polar.
    </Tip>
  </Step>

  <Step title="Choose a delivery format">
    For standard, custom integrations, leave this parameter on **Raw**. This will send a payload in JSON format.

    If you wish to send notifications to a Discord or Slack channel, you can select the corresponding format here. Polar will then adapt the payload so properly formatted messages are sent to your channel.

    <img className="block dark:hidden" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=77c7f9e6a2f2c3bd1bede381f692908c" width="1034" height="402" data-path="assets/integrate/webhooks/format.light.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=277936fd188888f6e4b81f8973f2f8ac 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=acded2e888ca6e24d95d4684793139a9 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=566b754e7651e74e8876e4fcb3692467 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=7e607f77f8b1811f5ae45a269fea7afb 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=ba2313fa067e67b0ec450a2d83643c49 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.light.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=c6e8087507a215a93b47d821c536ea47 2500w" data-optimize="true" data-opv="2" />

    <img className="hidden dark:block" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=896fc481fe7c61892b034380247479ae" width="1050" height="418" data-path="assets/integrate/webhooks/format.dark.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=46a709ed3ee3cc5c0e879e62eda39d87 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=473dd0c2d23a6018519bdc5db9198635 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=360f78fc430b955e0c7148c428c672a1 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=0182443ea5be4606ce19c61e383b0dbb 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=43763ad9655461dd11d8b0a427ac7d86 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/format.dark.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=d17c85b66413e5749f7681ad3075fda2 2500w" data-optimize="true" data-opv="2" />

    If you paste a Discord or Slack Webhook URL, the format will be automatically selected.
  </Step>

  <Step title="Set a secret">
    We cryptographically sign the requests using this secret. So you can easily
    verify them using our SDKs to ensure they are legitimate webhook payloads
    from Polar.

    You can set your own or generate a random one.

    <img className="block dark:hidden" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=e9fa8ce33d0e86d6331813f4a37ab509" width="1074" height="331" data-path="assets/integrate/webhooks/secret.light.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=4461f60b5496f5ebaabdaaeb21ec3277 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=0d849356d3b2b6fee04f64499211e30c 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=9930ebe92473176eb9609e7a2e519d9a 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=b973426f396f58c78ba1ed383ca48cd1 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=e1cf218d85849a7decda559d7a80bdb9 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.light.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=cce65883b68b70301736c88079002ee5 2500w" data-optimize="true" data-opv="2" />

    <img className="hidden dark:block" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=d447062a9726d8d9116aef2984a96cc2" width="1072" height="332" data-path="assets/integrate/webhooks/secret.dark.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=e95eaeac04c68cdf4ba4fe55bd3cb011 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=0254c3bbabf858a252aeac0d41a0980e 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=6e4bb6b660dbd7267967be16dbd683d4 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=6200bd4919d56b4d28a836d3ef2adcf5 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=5066f5e1100d89df0c798edb0e4dae2c 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/secret.dark.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=3e3435cd0dc0323f90420e6302f861e7 2500w" data-optimize="true" data-opv="2" />
  </Step>

  <Step title="Subscribe to events">
    Finally, select all the events you want to be notified about and you're done 🎉
  </Step>
</Steps>

[Now, it's time to integrate our endpoint to receive events
→](/integrate/webhooks/delivery)

# Handle & monitor webhook deliveries

> How to parse, validate and handle webhooks and monitor their deliveries on Polar

<img className="block dark:hidden" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=1c5f560bcca11607c67cc7f8b467dca4" width="2740" height="1522" data-path="assets/integrate/webhooks/delivery.light.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=61c5f66e8cb7acab698915171b5bde4e 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=f452e9e0edebfa7b912f85238d9d7f3c 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=49d406e3c294cb33ac178f7c61cd81aa 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=f2a6770fba4cc61303cc7d84de7db7bb 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=609d960dff8b2cf958e52118b7747365 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.light.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=683818fb66539342505f27574af611a5 2500w" data-optimize="true" data-opv="2" />

<img className="hidden dark:block" src="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=4ce4ac52095eb88c9073c4a18099a0bf" width="2672" height="1526" data-path="assets/integrate/webhooks/delivery.dark.png" srcset="https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?w=280&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=623e1747b6245fd84986462620718bcc 280w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?w=560&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=da4785f9e803826575e0a148286e3c43 560w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?w=840&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=0d3e453abcd767cddea396717bc774ec 840w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?w=1100&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=314fcf0248f280018d16f61ff50cb86c 1100w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?w=1650&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=aa74acbb1c73448868054df87bd7e0ad 1650w, https://mintcdn.com/polar/0Af3hN6-oIM4IHT3/assets/integrate/webhooks/delivery.dark.png?w=2500&fit=max&auto=format&n=0Af3hN6-oIM4IHT3&q=85&s=9117a6ffafa146e98eed1f1f1582a776 2500w" data-optimize="true" data-opv="2" />

Once a webhook endpoint is setup you will have access to the delivery overview
page. Here you can:

* See historic deliveries
* Review payload sent
* Trigger redelivery in case of failure

Now, let's integrate our endpoint route to validate, parse & handle incoming webhooks.

## Validate & parse webhooks

You now need to setup a route handler for the endpoint registered on Polar to
receive, validate and parse webhooks before handling them according to your
needs.

### Using our SDKs

Our TypeScript & Python SDKs come with a built-in helper function to easily
validate and parse the webhook event - see full examples below.

<CodeGroup>
  ```typescript JS (Express)
  import express, { Request, Response } from 'express'
  import { validateEvent, WebhookVerificationError } from '@polar-sh/sdk/webhooks'

  const app = express()

  app.post(
  '/webhook',
  express.raw({ type: 'application/json' }),
  (req: Request, res: Response) => {
  try {
  const event = validateEvent(
  req.body,
  req.headers,
  process.env['POLAR_WEBHOOK_SECRET'] ?? '',
  )

        // Process the event

        res.status(202).send('')
      } catch (error) {
        if (error instanceof WebhookVerificationError) {
          res.status(403).send('')
        }
        throw error
      }

  },
  )

  ```

  ```python Python (Flask)
  import os
  from flask import Flask, request
  from polar_sdk.webhooks import validate_event, WebhookVerificationError

  app = Flask(__name__)

  @app.route('/webhook', methods=['POST'])
  def webhook():
      try:
          event = validate_event(
              body=request.data,
              headers=request.headers,
              secret=os.getenv('POLAR_WEBHOOK_SECRET', ''),
          )

          # Process the event

          return "", 202
      except WebhookVerificationError as e:
          return "", 403
  ```
</CodeGroup>

Both examples above expect an environment variable named `POLAR_WEBHOOK_SECRET`
to be set to the secret you configured during the endpoint setup.

### Custom validation

We follow the [Standard Webhooks](https://www.standardwebhooks.com/)
standard which offers [many libraries across languages](https://github.com/standard-webhooks/standard-webhooks/tree/main/libraries) to
easily validate signatures. Or you can follow their
[specification](https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md)
in case you want to roll your own.

<Info>
  **Note: Secret needs to be base64 encoded**

  One common gotcha with the specification is that the webhook secret is expected to be
  base64 encoded. You don't have to do this with our SDK as it takes care of the
  implementation details with better developer ergonomics.
</Info>

## IP Allowlist

If you are using a firewall or a reverse proxy that requires IP allowlisting, here are the IPs you need to allow:

<CodeGroup>
  ```txt Production
  3.134.238.10
  3.129.111.220
  52.15.118.168
  ```

  ```txt Sandbox
  3.134.238.10
  3.129.111.220
  52.15.118.168
  ```
</CodeGroup>

## Failure Handling

### Delivery Retries

If we hit an error while trying to reach your endpoint, whether it is a temporary network error or a bug, we'll retry to send the event up to **10 times** with an exponential backoff.

### Delivery Timeouts

We timeout our requests to your endpoint after **20 seconds**. Triggering a
retry attempt after a delay as explained above. However, we strongly recommend you optimize your endpoint route to be fast. A
best practice is for your webhook handler to queue a background worker task to handle the
payload asynchronously.

## Troubleshooting

### Not receiving webhooks

Seeing deliveries on Polar, but not receiving them on your end? Below are some
common techniques to resolve the issue depending on the reported error status.

**General**

*Start ngrok or similar*

Make sure you have started `ngrok` or whatever tunneling service you're using
during local development.

*Add excessive logging*

E.g
`console.log('webhook.handler_called')`,
`console.log('webhook.validate_signature')`,
`console.log('webhook.signature_validated')` etc.

So you can easily confirm if the handler is called and how far it gets before
any issues arise.

`HTTP 404`

* Try `curl -vvv -X POST <copy-paste-endpoint-url>` in your terminal to confirm the
  route exists and see any issues along the way
* Try adding trailing `/` to the URL on Polar. Often `/foo` is resolved to
  `/foo/` by frameworks.

`HTTP 403`

* Using middleware for authorization? Make sure to exclude the webhook route
  from it since it needs to be publicly accessible
* Using Cloudflare?
  * Check the firewall logs to verify if they are blocking our requests and setup a custom WAF rule to accept incoming requests from Polar.
  * Webhook delivery failures with 403 errors can occur when Cloudflare's Bot Fight Mode is enabled. Bot Fight Mode automatically blocks requests it identifies as bots, including legitimate webhook requests from Polar. Adding Polar's IP addresses to your IP Allow List or creating custom WAF rules will not resolve this issue. To fix webhook delivery problems, disable Bot Fight Mode in your Cloudflare dashboard under Security > Bots. Alternatively, you can check your Cloudflare firewall logs to confirm if requests are being blocked and create appropriate firewall rules if needed.

### Invalid signature exceptions

Rolling your own webhook validation logic? Make sure to base64 encode the secret
you configured on Polar in your code before generating the signature to validate
against.

# Webhook Events

> Our webhook events and in which context they are useful

## Billing Events

### Checkout

<Card title="checkout.created" icon="link" href="/api-reference/webhooks/checkout.created" horizontal />

<Card title="checkout.updated" icon="link" href="/api-reference/webhooks/checkout.updated" horizontal />

### Customers

<Card title="customer.created" icon="link" href="/api-reference/webhooks/customer.created" horizontal>
  Fired when a new customer has been created.
</Card>

<Card title="customer.updated" icon="link" href="/api-reference/webhooks/customer.updated" horizontal>
  Fired when a customer has been updated.
</Card>

<Card title="customer.deleted" icon="link" href="/api-reference/webhooks/customer.deleted" horizontal>
  Fired when a customer has been deleted.
</Card>

<Card title="customer.state_changed" icon="link" href="/api-reference/webhooks/customer.state_changed" horizontal>
  Fired when a customer's state has changed. Includes active subscriptions and
  granted benefits.
</Card>

### Subscriptions

In order to properly implement logic for handling subscriptions, you should look into the following events.

<Card title="subscription.created" icon="link" href="/api-reference/webhooks/subscription.created" horizontal>
  Fired when a new subscription has been created.
</Card>

<Card title="subscription.updated" icon="link" href="/api-reference/webhooks/subscription.updated" horizontal>
  Use this event if you want to handle cancellations, un-cancellations, etc. The
  updated event is a catch-all event for `subscription.active` ,
  `subscription.canceled`, `subscription.uncanceled` and `subscription.revoked`.
</Card>

<Card title="order.created" icon="link" href="/api-reference/webhooks/order.created" horizontal>
  In case you want to do logic when a subscription is renewed, you should listen
  to `order.created` and the `billing_reason` field. It can be `purchase`,
  `subscription_create`, `subscription_cycle` and `subscription_update`.
  `subscription_cycle` is used when subscriptions renew.
</Card>

<Card title="subscription.active" icon="link" href="/api-reference/webhooks/subscription.active" horizontal />

<Card title="subscription.canceled" icon="link" href="/api-reference/webhooks/subscription.canceled" horizontal />

<Card title="subscription.uncanceled" icon="link" href="/api-reference/webhooks/subscription.uncanceled" horizontal />

<Card title="subscription.revoked" icon="link" href="/api-reference/webhooks/subscription.revoked" horizontal />

#### Cancellation Sequences

When a subscription is canceled, the events triggered depend on whether the cancellation is immediate or scheduled for the end of the billing period.

**End-of-Period Cancellation (default)**

When a subscription is **canceled** (by customer action from the portal or by the merchant from the dashboard/API), the following events are sent immediately:

1. `subscription.updated`
2. `subscription.canceled`

Both events contain the same subscription data. The subscription will still have `active` status, but the `cancel_at_period_end` flag will be set to `true`.

When the end of the current billing period arrives, the subscription is definitively revoked: billing cycles stop and benefits are revoked. The following events are then sent:

3. `subscription.updated`
4. `subscription.revoked`

Both events contain the same subscription data. The subscription will have the `canceled` status.

**Immediate Revocation**

When a merchant cancels a subscription with **immediate revocation**, those events are sent at once:

1. `subscription.updated`
2. `subscription.canceled`
3. `subscription.revoked`

All three events contain the same subscription data. The subscription will have the `canceled` status immediately.

#### Renewal Sequences

When a subscription is renewed for a new cycle, the webhook events are triggered in a specific sequence to help you track the renewal process and handle billing logic appropriately.

**Initial Renewal Events**

When a subscription reaches its renewal date, the following events are sent immediately (if enabled on the webhook):

1. `subscription.updated`
2. `order.created`

The subscription data will reflect the new billing period through the `current_period_start` and `current_period_end` properties, showing the updated cycle dates.

The order data represents the new invoice for the upcoming cycle, with a total representing what the customer will pay for this new period. If usage-based billing is involved, their consumption for the past period will be included in the total. The status of this order is `pending` at this stage.

**Payment Processing Events**

Shortly after the initial renewal events, the platform will trigger a payment for the new order. Once the payment is successfully processed, the following events are sent:

3. `order.updated`
4. `order.paid`

Both events will contain the same order data, with the order status changed to `paid`.

### Order

<Card title="order.created" icon="link" href="/api-reference/webhooks/order.created" horizontal />

<Card title="order.paid" icon="link" href="/api-reference/webhooks/order.paid" horizontal />

<Card title="order.updated" icon="link" href="/api-reference/webhooks/order.updated" horizontal />

<Card title="order.refunded" icon="link" href="/api-reference/webhooks/order.refunded" horizontal />

### Refunds

<Card title="refund.created" icon="link" href="/api-reference/webhooks/refund.created" horizontal />

<Card title="refund.updated" icon="link" href="/api-reference/webhooks/refund.updated" horizontal />

### Benefit Grants

<Card title="benefit_grant.created" icon="link" href="/api-reference/webhooks/benefit_grant.created" horizontal />

<Card title="benefit_grant.updated" icon="link" href="/api-reference/webhooks/benefit_grant.updated" horizontal />

<Card title="benefit_grant.revoked" icon="link" href="/api-reference/webhooks/benefit_grant.revoked" horizontal />

## Organization Events

### Benefits

<Card title="benefit.created" icon="link" href="/api-reference/webhooks/benefit.created" horizontal />

<Card title="benefit.updated" icon="link" href="/api-reference/webhooks/benefit.updated" horizontal />

### Products

<Card title="product.created" icon="link" href="/api-reference/webhooks/product.created" horizontal />

<Card title="product.updated" icon="link" href="/api-reference/webhooks/product.updated" horizontal />

### Organization

<Card title="organization.updated" icon="link" href="/api-reference/webhooks/organization.updated" horizontal />

