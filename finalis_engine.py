"""
FINALIS CONTRACT PROCESSING ENGINE
Python implementation - Testing
VERSION 2.0 
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
import json


class FinalisEngine:
    """
    Processes M&A deals according to Finalis contract rules.
    All calculations are deterministic and precise.
    """

    def __init__(self):
        self.FINRA_RATE = Decimal('0.004732')
        self.DISTRIBUTION_RATE = Decimal('0.10')
        self.SOURCING_RATE = Decimal('0.10')
        self.DEAL_EXEMPT_RATE = Decimal('0.015')

    @staticmethod
    def to_money(value: Decimal) -> float:
        """Convert Decimal to float with exactly 2 decimal places"""
        return round(float(value), 2)

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data to catch errors early.
        Raises ValueError with clear messages if input is invalid.
        """
        state = input_data.get('state', {})
        contract = input_data.get('contract', {})
        new_deal = input_data.get('new_deal', {})

        # Validate deal
        if 'success_fees' not in new_deal:
            raise ValueError("Missing required field: new_deal.success_fees")

        success_fees = Decimal(str(new_deal['success_fees']))
        external_retainer = Decimal(str(new_deal.get('external_retainer', 0)))

        if success_fees <= 0:
            raise ValueError(
                f"success_fees must be positive, got: {success_fees}")


        if external_retainer < 0:
            raise ValueError(
                f"external_retainer cannot be negative, got: {external_retainer}"
            )

        # Validate external retainer type
        has_external_retainer = new_deal.get('has_external_retainer', False)
        if has_external_retainer:
            is_deducted = new_deal.get('is_external_retainer_deducted')
            if is_deducted is None:
                raise ValueError(
                    "is_external_retainer_deducted is required when has_external_retainer=True"
                )

            # Validate that external_retainer amount is provided
            if external_retainer <= 0:
                raise ValueError(
                    f"external_retainer must be positive when has_external_retainer=True, got: {external_retainer}"
                )

        # Validate state
        current_credit = Decimal(str(state.get('current_credit', 0)))
        current_debt = Decimal(str(state.get('current_debt', 0)))

        if current_credit < 0:
            raise ValueError(
                f"current_credit cannot be negative, got: {current_credit}")
        if current_debt < 0:
            raise ValueError(
                f"current_debt cannot be negative, got: {current_debt}")

        # Validate future payments
        for payment in state.get('future_subscription_fees', []):
            amount_due = Decimal(str(payment.get('amount_due', 0)))
            amount_paid = Decimal(str(payment.get('amount_paid', 0)))

            if amount_due < 0:
                raise ValueError(f"amount_due cannot be negative: {payment}")
            if amount_paid < 0:
                raise ValueError(f"amount_paid cannot be negative: {payment}")
            if amount_paid > amount_due:
                raise ValueError(
                    f"amount_paid cannot exceed amount_due: {payment}")

        # Validate contract
        rate_type = contract.get('rate_type')
        if rate_type not in ['fixed', 'lehman']:
            raise ValueError(
                f"Invalid rate_type: {rate_type}. Must be 'fixed' or 'lehman'")

        if rate_type == 'fixed':
            fixed_rate = contract.get('fixed_rate')
            if fixed_rate is None:
                raise ValueError(
                    "fixed_rate is required when rate_type='fixed'")
            if not (0 <= fixed_rate <= 1):
                raise ValueError(
                    f"fixed_rate must be between 0 and 1, got: {fixed_rate}")

        if rate_type == 'lehman':
            lehman_tiers = contract.get('lehman_tiers')
            if not lehman_tiers:
                raise ValueError(
                    "lehman_tiers is required when rate_type='lehman'")

            # Validate tiers don't overlap
            for i, tier in enumerate(lehman_tiers):
                if 'lower_bound' not in tier or 'rate' not in tier:
                    raise ValueError(
                        f"Tier {i} missing required fields: {tier}")

                rate = Decimal(str(tier['rate']))
                if not (0 <= rate <= 1):
                    raise ValueError(
                        f"Tier {i} rate must be between 0 and 1, got: {rate}")

            # Validate preferred rate (if present)
            has_preferred_rate = new_deal.get('has_preferred_rate', False)
            if has_preferred_rate:
                preferred_rate = new_deal.get('preferred_rate')
                if preferred_rate is None:
                    raise ValueError(
                        "preferred_rate is required when has_preferred_rate=True")

                preferred_rate_decimal = Decimal(str(preferred_rate))
                if not (0 <= preferred_rate_decimal <= 1):
                    raise ValueError(
                        f"preferred_rate must be between 0 and 1, got: {preferred_rate}")
            

            # Validate PAYG constraints
            is_payg = contract.get('is_pay_as_you_go', False)
            if is_payg:
                if current_credit > 0:
                    raise ValueError(
                        "Pay-As-You-Go contracts cannot have credit. PAYG contracts have no credit system."
                    )
                if len(state.get('future_subscription_fees', [])) > 0:
                    raise ValueError(
                        "Pay-As-You-Go contracts cannot have future subscription fees. PAYG has no subscription prepayments."
                    )

    def calculate_contract_year(self, contract_start_date: str,
                                deal_date: str) -> int:
        """
        Calculate which contract year we're in based on dates.
        Year 1 = first 12 months, Year 2 = next 12 months, etc.
        """
        from datetime import datetime

        start = datetime.strptime(contract_start_date, "%Y-%m-%d")
        deal = datetime.strptime(deal_date, "%Y-%m-%d")

        # Calculate difference in days
        days_diff = (deal - start).days

        # Calculate year (1-indexed)
        # Year 1: days 0-364, Year 2: days 365-729, etc.
        contract_year = (days_diff // 365) + 1

        return contract_year


    def calculate_applicable_deferred(self, state: Dict[str, Any], 
          contract_start_date: str,
          deal_date: str) -> Decimal:
        """
        Calculate which deferred amount applies based on the current contract year.
    
        LOGIC:
        - If deferred_schedule exists: use the amount for the current contract year
        - If deferred_subscription_fee exists (legacy): use that
        - Otherwise: return 0
    
        Returns: Decimal amount of deferred applicable to this deal
        """

        # Check if multi-year deferred schedule exists (NEW)
        deferred_schedule = state.get('deferred_schedule')

        if deferred_schedule and len(deferred_schedule) > 0:
        # Calculate current contract year
            current_year = self.calculate_contract_year(contract_start_date, deal_date)

            # Find deferred for this specific year
            for deferred_entry in deferred_schedule:
                if deferred_entry.get('year') == current_year:
                    amount = Decimal(str(deferred_entry.get('amount', 0)))  # ← DEFINIR PRIMERO
                    return amount  # ← Usar la variable

            # If no deferred for this year, return 0
            return Decimal('0')

        # Fallback to legacy single deferred (backward compatible)
        legacy_deferred = state.get('deferred_subscription_fee', 0)
        return Decimal(str(legacy_deferred))
    


    

    def apply_cost_cap(self, finalis_commissions: Decimal, 
           implied_total_before_subscription: Decimal,  # NUEVO parámetro
           contract: Dict[str, Any],
           state: Dict[str, Any], 
           advance_fees_created: Decimal,
           deal_date: str) -> tuple[Decimal, Decimal]:
        """
        Apply cost cap limits to finalis commissions.
    
        LÓGICA:
        - Cost Cap puede ser: "annual" (por año), "total" (todo el contrato), o None
        - Si es "annual": compara contra total_paid_this_contract_year
        - Si es "total": compara contra total_paid_all_time
        - Advance fees tienen prioridad, commissions se limitan si exceden el cap
    
        IMPORTANTE PARA PAYG:
        - amount_not_charged debe calcularse desde el implied_total ORIGINAL (antes de restar subscription)
    
        Returns: (finalis_commissions_after_cap, amount_not_charged)
        """
        # Check if cost cap exists
        cost_cap_type = contract.get('cost_cap_type')  # "annual", "total", or None
        cost_cap_amount = contract.get('cost_cap_amount')
    
        if cost_cap_type is None or cost_cap_amount is None:
        # No cost cap - return original amount
            return finalis_commissions, Decimal('0')
    
        cost_cap_amount_decimal = Decimal(str(cost_cap_amount))
    
        # Get the appropriate tracking amount based on cap type
        if cost_cap_type == "annual":
            total_paid = Decimal(str(state.get('total_paid_this_contract_year', 0)))
        elif cost_cap_type == "total":
            total_paid = Decimal(str(state.get('total_paid_all_time', 0)))
        else:
            # Invalid cap type - no cap applies
            return finalis_commissions, Decimal('0')
    
        # Calculate available space under cap
        available_space = max(Decimal('0'), cost_cap_amount_decimal - total_paid)
    
        # Calculate total amount we want to charge this deal
        total_to_charge_this_deal = advance_fees_created + finalis_commissions
    
        if total_to_charge_this_deal <= available_space:
        # Everything fits within the cap
            commissions_after_cap = finalis_commissions
            amount_not_charged = Decimal('0')
        else:
            # We exceed the cap
            # Advance fees have priority (already created)
            # So we limit only the commissions
            space_for_commissions = max(Decimal('0'), available_space - advance_fees_created)
            commissions_after_cap = min(finalis_commissions, space_for_commissions)
    
        
            amount_not_charged = implied_total_before_subscription - (advance_fees_created + commissions_after_cap)
    
        
            amount_not_charged = max(Decimal('0'), amount_not_charged)

        return commissions_after_cap, amount_not_charged

    def process_deal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing function - executes all steps in order.
        """
        # Validate input first
        self.validate_input(input_data)

        state = input_data['state']
        contract = input_data['contract']
        new_deal = input_data['new_deal']

        # Convert to Decimal for precise calculations
        success_fees = Decimal(str(new_deal['success_fees']))
        external_retainer = Decimal(str(new_deal.get('external_retainer', 0)))
        has_external_retainer = new_deal.get('has_external_retainer', False)
        is_external_retainer_deducted = new_deal.get('is_external_retainer_deducted', True)

        # Calculate total based on retainer type
        if has_external_retainer and is_external_retainer_deducted:
            # TYPE 1: Retainer is part of success fee (deducted)
            total_for_calculations = success_fees + external_retainer
        elif has_external_retainer and not is_external_retainer_deducted:
            # TYPE 2: Retainer is standalone (not part of success fee)
            total_for_calculations = success_fees  # Ignore external retainer for calculations
        else:
            # No external retainer
            total_for_calculations = success_fees

        current_credit = Decimal(str(state['current_credit']))
        current_debt = Decimal(str(state['current_debt']))

        # STEP 1: Calculate Fixed Costs
        has_finra_fee = new_deal.get('has_finra_fee', True)  # Default True (backward compatible)
        finra_fee = self.calculate_finra_fee(total_for_calculations, has_finra_fee)
        distribution_fee = self.calculate_distribution_fee(
            total_for_calculations, new_deal['is_distribution_fee_true'])
        sourcing_fee = self.calculate_sourcing_fee(
            total_for_calculations, new_deal['is_sourcing_fee_true'])

        # STEP 2: Calculate IMPLIED (BD Cost)
        accumulated_before = Decimal(
            str(contract['accumulated_success_fees_before_this_deal']))

        implied_total = self.calculate_implied(
            total_for_calculations,
            new_deal['is_deal_exempt'],
            contract['rate_type'],
            contract.get('fixed_rate'),
            contract.get('lehman_tiers'),
            accumulated_before,
            new_deal.get('has_preferred_rate', False),  # NUEVO
            new_deal.get('preferred_rate')              # NUEVO
        )

        # Get contract type early (needed for credit logic)
        is_payg = contract.get('is_pay_as_you_go', False)

        # STEP 3: Process Debt Collection (including deferred backend)
        # NEW: Calculate which deferred applies based on contract year
        deferred_backend = self.calculate_applicable_deferred(
            state, 
            contract.get('contract_start_date'),
            new_deal['deal_date']
        )
        total_debt = current_debt + deferred_backend
        debt_collected = min(success_fees, total_debt)


        # Split collected debt between regular debt and deferred
        if debt_collected > 0:
            if current_debt > 0:
                regular_debt_collected = min(debt_collected, current_debt)
                deferred_collected = debt_collected - regular_debt_collected
            else:
                regular_debt_collected = Decimal('0')
                deferred_collected = debt_collected
        else:
            regular_debt_collected = Decimal('0')
            deferred_collected = Decimal('0')

        remaining_debt = current_debt - regular_debt_collected
        remaining_deferred = deferred_backend - deferred_collected

        # STEP 4: Convert Collected Debt to Credit
        # CRITICAL: Both REGULAR debt AND DEFERRED generate credit (both come from gross)
        # CRITICAL: PAYG contracts do NOT generate or use credit
        if is_payg:
            # Pay-As-You-Go: NO credit system
            credit_from_debt = Decimal('0')
            new_credit = Decimal('0')  # PAYG has no credit
        else:
            # Standard contracts: ALL debt collected generates credit
            credit_from_debt = debt_collected  # ← CHANGED: total debt collected (regular + deferred)
            new_credit = current_credit + credit_from_debt

        # STEP 5: Apply Credit Against IMPLIED
        is_in_commissions_mode = state['is_in_commissions_mode']

        if is_payg:
            # Pay-As-You-Go: NO usa credit
            credit_used_for_implied = Decimal('0')
            credit_remaining = new_credit
            implied_remaining_after_credit = implied_total
        elif is_in_commissions_mode:
            # Already in commissions mode - credit is NOT used
            credit_used_for_implied = Decimal('0')
            credit_remaining = new_credit
            implied_remaining_after_credit = implied_total
        else:
            # Normal case - credit absorbs implied
            credit_used_for_implied = min(implied_total, new_credit)
            credit_remaining = new_credit - credit_used_for_implied
            implied_remaining_after_credit = implied_total - credit_used_for_implied

        # STEP 6: Create Advance Subscription Fees
        future_fees = state['future_subscription_fees']

        if is_payg:
            # Pay-As-You-Go: NO crea advance fees
            advance_fees_created = Decimal('0')
            updated_payments = []
            contract_fully_prepaid = True  # PAYG siempre está "prepaid"
        else:
            # Standard contracts: create advance fees normally
            advance_fees_created, updated_payments, contract_fully_prepaid = self.process_advance_fees(
                implied_remaining_after_credit, future_fees)

        # STEP 7: Calculate Finalis Commissions
        implied_remaining_after_advance = implied_remaining_after_credit - advance_fees_created

        if is_payg:
            # Pay-As-You-Go: Check if ARR is already covered
            arr = Decimal(str(contract.get('annual_subscription', 0)))
            payg_accumulated = Decimal(str(state.get('payg_commissions_accumulated', 0)))

            # Calculate how much of THIS deal's implied goes to ARR vs Finalis
            remaining_arr = max(Decimal('0'), arr - payg_accumulated)

            if implied_total <= remaining_arr:
                # All implied goes to ARR (not Finalis yet)
                finalis_commissions_before_cap = Decimal('0')
                new_commissions_mode = False
                entered_commissions_mode = False
            elif payg_accumulated >= arr:
                # ARR already covered - all implied becomes Finalis commissions
                finalis_commissions_before_cap = implied_total
                new_commissions_mode = True
                entered_commissions_mode = False
            else:
                # Partial: some to ARR, rest to Finalis
                finalis_commissions_before_cap = implied_total - remaining_arr
                new_commissions_mode = True
                entered_commissions_mode = True  # Just entered commissions mode

        elif is_in_commissions_mode:
            # CASE 1: Already in commissions mode
            finalis_commissions_before_cap = implied_total
            new_commissions_mode = True
            entered_commissions_mode = False
        elif contract_fully_prepaid:
            # CASE 2: Became fully prepaid NOW
            finalis_commissions_before_cap = implied_remaining_after_advance
            new_commissions_mode = True
            entered_commissions_mode = True
        else:
            # CASE 3: Not fully prepaid
            finalis_commissions_before_cap = Decimal('0')
            new_commissions_mode = False
            entered_commissions_mode = False

        # STEP 7b: Apply Cost Cap (if exists)
        finalis_commissions, amount_not_charged_due_to_cap = self.apply_cost_cap(
            finalis_commissions_before_cap, 
            implied_total,  # ← NUEVO: pasar el implied_total original
            contract, 
            state,
            advance_fees_created, 
            new_deal['deal_date'])

        # STEP 8: Calculate Net Payout
        net_payout = self.calculate_net_payout(success_fees, debt_collected,
                                               finra_fee, distribution_fee,
                                               sourcing_fee,
                                               advance_fees_created,
                                               finalis_commissions)

        # STEP 9: Construct Updated State
        updated_accumulated = accumulated_before + total_for_calculations

        # Build output - all monetary values with exactly 2 decimals
        result = {
            "deal_summary": {
                "deal_name": new_deal['deal_name'],
                "success_fees": self.to_money(success_fees),
                "external_retainer": self.to_money(external_retainer),
                "external_retainer_deducted": is_external_retainer_deducted if has_external_retainer else None,
                "total_deal_value": self.to_money(total_for_calculations),
                "deal_date": new_deal['deal_date'],
                "has_finra_fee": has_finra_fee  

            },
            "calculations": {
                "finra_fee":
                self.to_money(finra_fee),
                "distribution_fee":
                self.to_money(distribution_fee),
                "sourcing_fee":
                self.to_money(sourcing_fee),
                "implied_total":
                self.to_money(implied_total),
                "debt_collected":
                self.to_money(debt_collected),
                "deferred_collected":
                self.to_money(deferred_collected),
                "credit_used_for_implied":
                self.to_money(credit_used_for_implied),
                "advance_fees_created":
                self.to_money(advance_fees_created),
                "finalis_commissions":
                self.to_money(finalis_commissions),
                "finalis_commissions_before_cap":
                self.to_money(finalis_commissions_before_cap),
                "amount_not_charged_due_to_cap":
                self.to_money(amount_not_charged_due_to_cap),
                "net_payout_to_client":
                self.to_money(net_payout)
            },
            "state_changes": {
                "initial_credit": self.to_money(current_credit),
                "final_credit": self.to_money(credit_remaining),
                "initial_debt": self.to_money(current_debt),
                "final_debt": self.to_money(remaining_debt),
                "initial_deferred": self.to_money(deferred_backend),
                "final_deferred": self.to_money(remaining_deferred),
                "contract_year": self.calculate_contract_year(
                    contract.get('contract_start_date'), 
                    new_deal['deal_date']
                ) if contract.get('contract_start_date') else None,
                "contract_fully_prepaid": contract_fully_prepaid,
                "entered_commissions_mode": entered_commissions_mode
            },
            "updated_future_payments": updated_payments,
            "updated_contract_state": {
                "current_credit":
                self.to_money(credit_remaining),
                "current_debt":
                self.to_money(remaining_debt),
                "deferred_subscription_fee":
                self.to_money(remaining_deferred),
                "is_in_commissions_mode":
                new_commissions_mode,
                "accumulated_success_fees":
                self.to_money(updated_accumulated),
                "total_paid_this_contract_year":
                self.to_money(
                    Decimal(str(state.get('total_paid_this_contract_year', 0)))
                    + advance_fees_created + finalis_commissions),
                "total_paid_all_time":
                self.to_money(
                    Decimal(str(state.get('total_paid_all_time', 0))) +
                    advance_fees_created + finalis_commissions)
            }
        }

        # Add PAYG tracking if applicable
        if is_payg:
            arr = Decimal(str(contract.get('annual_subscription', 0)))
            payg_commissions_paid = Decimal(
                str(state.get('payg_commissions_accumulated',
                              0))) + finalis_commissions

            # Determine how much is ARR Commissions vs Finalis Commissions
            if payg_commissions_paid <= arr:
                # All commissions are still covering ARR
                arr_commissions = finalis_commissions
                finalis_commissions_excess = Decimal('0')
            else:
                # Some commissions exceed ARR
                previous_paid = Decimal(
                    str(state.get('payg_commissions_accumulated', 0)))
                if previous_paid >= arr:
                    # Already covered ARR completely - all is excess
                    arr_commissions = Decimal('0')
                    finalis_commissions_excess = finalis_commissions
                else:
                    # Partially covering ARR, rest is excess
                    remaining_arr = arr - previous_paid
                    arr_commissions = min(finalis_commissions, remaining_arr)
                    finalis_commissions_excess = finalis_commissions - arr_commissions

            result['payg_tracking'] = {
                "arr_target":
                self.to_money(arr),
                "commissions_accumulated":
                self.to_money(payg_commissions_paid),
                "remaining_to_cover_arr":
                self.to_money(max(Decimal('0'), arr - payg_commissions_paid)),
                "arr_coverage_percentage":
                round(
                    float(
                        (payg_commissions_paid / arr * 100)) if arr > 0 else 0,
                    2),
                "arr_commissions":
                self.to_money(arr_commissions),
                "finalis_commissions_excess":
                self.to_money(finalis_commissions_excess)
            }

            result['updated_contract_state'][
                'payg_commissions_accumulated'] = self.to_money(
                    payg_commissions_paid)

        # NEW: If deferred_schedule exists, update it
        if state.get('deferred_schedule'):
            current_year = self.calculate_contract_year(
                contract.get('contract_start_date'),
                new_deal['deal_date']
            )

            updated_schedule = []
            for entry in state['deferred_schedule']:
                if entry['year'] == current_year:
                    # Update this year's deferred with remaining amount
                    updated_schedule.append({
                        "year": entry['year'],
                        "amount": self.to_money(remaining_deferred)
                    })
                else:
                    # Keep other years unchanged
                    updated_schedule.append({
                        "year": entry['year'],
                        "amount": entry['amount']
                    })

            result['updated_contract_state']['deferred_schedule'] = updated_schedule

        return result

    def calculate_finra_fee(self, success_fees: Decimal, has_finra_fee: bool = True) -> Decimal:
        """
        STEP 1: Calculate FINRA/SIPC fee (0.4732%)

        Args:
            success_fees: Deal success fees
            has_finra_fee: Whether FINRA fee applies (default True for backward compatibility)

        Returns:
            FINRA fee amount (0 if exempt)
        """
        if not has_finra_fee:
            return Decimal('0')

        return (success_fees * self.FINRA_RATE).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP)

    def calculate_distribution_fee(self, success_fees: Decimal,
                                   is_true: bool) -> Decimal:
        """STEP 1: Calculate distribution fee (10% if applicable)"""
        if is_true:
            # FIX: Add quantize to avoid floating precision issues
            return (success_fees * self.DISTRIBUTION_RATE).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)
        return Decimal('0')

    def calculate_sourcing_fee(self, success_fees: Decimal,
                               is_true: bool) -> Decimal:
        """STEP 1: Calculate sourcing fee (10% if applicable)"""
        if is_true:
            # FIX: Add quantize to avoid floating precision issues
            return (success_fees * self.SOURCING_RATE).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)
        return Decimal('0')

    def calculate_implied(self, success_fees: Decimal, is_deal_exempt: bool,
          rate_type: str, fixed_rate: Optional[float],
          lehman_tiers: Optional[List[Dict]],
          accumulated_before: Decimal,
          has_preferred_rate: bool = False,          # NUEVO
          preferred_rate: Optional[float] = None    # NUEVO
          ) -> Decimal:
        """
        STEP 2: Calculate IMPLIED (BD Cost)
        Four modes (in priority order):
        1. Preferred Rate (deal-specific override)
        2. Deal Exempt
        3. Lehman Progressive Tiers
        4. Fixed Rate
        """
        # MODE 1: Preferred Rate (HIGHEST PRIORITY)
        if has_preferred_rate and preferred_rate is not None:
            return (success_fees * Decimal(str(preferred_rate))).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)

        # MODE 2: Deal Exempt
        if is_deal_exempt:
            return (success_fees * self.DEAL_EXEMPT_RATE).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)

    # MODE 3: Lehman Progressive Tiers
        if rate_type == 'lehman' and lehman_tiers:
            return self.calculate_lehman_implied(success_fees, lehman_tiers,
                                 accumulated_before)

    # MODE 4: Fixed Rate
        if fixed_rate is not None:
            return (success_fees * Decimal(str(fixed_rate))).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)

        raise ValueError("Invalid rate configuration")


    

    def calculate_lehman_implied(self, success_fees: Decimal,
                                 lehman_tiers: List[Dict],
                                 accumulated_before: Decimal) -> Decimal:
        """
        Calculate implied using Lehman progressive tiers.
        Accounts for historical production.
        FIX: Handles gaps between tiers (e.g., tier ends at 100k, next starts at 100k.01)
        FIX: accumulated_before is now Decimal
        """
        acc = accumulated_before
        deal_amount = success_fees
        implied = Decimal('0')

        for tier in lehman_tiers:
            lower = Decimal(str(tier['lower_bound']))
            rate = Decimal(str(tier['rate']))

            # Determine tier capacity
            if tier['upper_bound'] is None:
                upper = None
            else:
                upper = Decimal(str(tier['upper_bound']))

            # FIX: Handle gaps between tiers
            if acc < lower:
                # We have a gap - jump to the start of this tier
                gap = lower - acc
                if gap <= deal_amount:
                    # We can reach this tier
                    deal_amount -= gap
                    acc = lower
                else:
                    # Deal ends before reaching this tier
                    break

            # Calculate how much of this tier was already consumed
            if upper is None:
                # Infinite tier
                used_in_tier = Decimal('0')
                remaining_capacity = deal_amount
            else:
                used_in_tier = max(Decimal('0'), acc - lower)
                tier_size = upper - lower
                remaining_capacity = max(Decimal('0'),
                                         tier_size - used_in_tier)

            # Allocate part of the deal to this tier
            allocated = min(remaining_capacity, deal_amount)

            # Calculate commission for this allocation
            tier_commission = (allocated * rate).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)
            implied += tier_commission

            # Update state
            deal_amount -= allocated
            acc += allocated

            # Stop if deal fully allocated
            if deal_amount == 0:
                break

        return implied

    def process_advance_fees(
            self, implied_remaining: Decimal,
            future_fees: List[Dict]) -> tuple[Decimal, List[Dict], bool]:
        """
        STEP 6: Create advance subscription fees.
        Returns: (advance_fees_created, updated_payments, contract_fully_prepaid)
        FIX: Use Decimal throughout to avoid rounding errors
        """
        if implied_remaining == 0:
            # No advance fees needed
            contract_fully_prepaid = len(future_fees) == 0
            updated_payments = [{
                "payment_id":
                p['payment_id'],
                "due_date":
                p['due_date'],
                "original_amount":
                self.to_money(Decimal(str(p['amount_due']))),
                "amount_paid":
                self.to_money(Decimal(str(p['amount_paid']))),
                "remaining":
                self.to_money(
                    Decimal(str(p['amount_due'])) -
                    Decimal(str(p['amount_paid'])))
            } for p in future_fees]
            return Decimal('0.00'), updated_payments, contract_fully_prepaid

        # Calculate total future fees owed (keep as Decimal)
        total_future_owed = sum(
            Decimal(str(p['amount_due'])) - Decimal(str(p['amount_paid']))
            for p in future_fees)

        # Advance fees cannot exceed total owed
        advance_fees_created = min(implied_remaining, total_future_owed)
        # Ensure it's Decimal
        if not isinstance(advance_fees_created, Decimal):
            advance_fees_created = Decimal(str(advance_fees_created))

        # Apply advance fees to payments in chronological order
        remaining_advance = advance_fees_created
        updated_payments = []

        # Sort by due_date
        sorted_fees = sorted(future_fees, key=lambda x: x['due_date'])

        for payment in sorted_fees:
            amount_due = Decimal(str(payment['amount_due']))
            amount_paid = Decimal(str(payment['amount_paid']))
            amount_owed = amount_due - amount_paid

            if remaining_advance >= amount_owed:
                # Fully cover this payment
                new_amount_paid = amount_paid + amount_owed
                remaining_advance -= amount_owed
            elif remaining_advance > 0:
                # Partially cover this payment
                new_amount_paid = amount_paid + remaining_advance
                remaining_advance = Decimal('0')
            else:
                # No advance left
                new_amount_paid = amount_paid

            remaining = amount_due - new_amount_paid

            updated_payments.append({
                "payment_id":
                payment['payment_id'],
                "due_date":
                payment['due_date'],
                "original_amount":
                self.to_money(amount_due),
                "amount_paid":
                self.to_money(new_amount_paid),
                "remaining":
                self.to_money(remaining)
            })

        # Check if contract is fully prepaid
        # FIX: Use Decimal comparison to avoid float precision issues
        contract_fully_prepaid = all(
            Decimal(str(p['remaining'])) == Decimal('0')
            for p in updated_payments)

        # Special case: empty future_subscription_fees
        if len(future_fees) == 0:
            contract_fully_prepaid = True

        return advance_fees_created, updated_payments, contract_fully_prepaid

    def calculate_net_payout(self, success_fees: Decimal,
                             debt_collected: Decimal, finra_fee: Decimal,
                             distribution_fee: Decimal, sourcing_fee: Decimal,
                             advance_fees: Decimal,
                             finalis_commissions: Decimal) -> Decimal:
        """
        STEP 8: Calculate net payout to client.
        IMPLIED is NEVER deducted (it was absorbed by credit or became advance/commissions).
        """
        net_payout = success_fees
        net_payout -= debt_collected
        net_payout -= finra_fee
        net_payout -= distribution_fee
        net_payout -= sourcing_fee
        net_payout -= advance_fees
        net_payout -= finalis_commissions

        return net_payout.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


# ============================================================================
# API FUNCTIONS
# ============================================================================


def process_deal_from_json(json_input: str) -> str:
    """
    Process a deal from JSON string input and return JSON string output.
    This is the function that will be called by the API.
    """
    try:
        # Parse input JSON
        input_data = json.loads(json_input)

        # Process the deal
        engine = FinalisEngine()
        result = engine.process_deal(input_data)

        # Return result as JSON string
        return json.dumps(result, indent=2)

    except ValueError as e:
        # Return validation error
        error_response = {"error": str(e), "status": "validation_failed"}
        return json.dumps(error_response, indent=2)

    except Exception as e:
        # Return other errors
        error_response = {"error": str(e), "status": "failed"}
        return json.dumps(error_response, indent=2)


def process_deal_from_dict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a deal from Python dict and return Python dict.
    Useful for programmatic usage.
    """
    engine = FinalisEngine()
    return engine.process_deal(input_data)
