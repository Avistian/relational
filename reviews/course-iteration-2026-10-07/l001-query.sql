WITH order_features AS (
  SELECT customer_id, COUNT(*) AS order_count_90d
  FROM orders
  WHERE order_date > date(:cutoff, '-90 days')
    AND order_date <= :cutoff
  GROUP BY customer_id
), session_features AS (
  SELECT customer_id, AVG(pages_viewed) AS avg_pages
  FROM sessions
  WHERE start_time <= :cutoff
  GROUP BY customer_id
), ticket_features AS (
  SELECT customer_id, COUNT(*) AS ticket_count
  FROM support_tickets
  WHERE opened_at <= :cutoff
  GROUP BY customer_id
)
SELECT c.customer_id,
       CAST(julianday(:cutoff) - julianday(c.signup_date)
            AS INTEGER) AS tenure,
       COALESCE(o.order_count_90d, 0) AS order_count_90d,
       s.avg_pages,
       COALESCE(t.ticket_count, 0) AS ticket_count
FROM customers AS c
LEFT JOIN order_features AS o USING (customer_id)
LEFT JOIN session_features AS s USING (customer_id)
LEFT JOIN ticket_features AS t USING (customer_id)
WHERE c.signup_date <= :cutoff;
