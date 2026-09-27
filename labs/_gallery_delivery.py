"""Reveal manifest navigation through the same native year disclosure as a learner."""
def reveal_gallery_link(page, selector):
    link = page.locator(selector).first
    link.wait_for(state='attached')
    ancestors = link.locator('xpath=ancestor::details')
    for i in range(ancestors.count()):
        group = ancestors.nth(i)
        if group.get_attribute('open') is None:
            group.locator('summary').first.click()
    link.wait_for(state='visible')
    return link
